#!/usr/bin/env python3
"""Record real-robot teleoperation episodes using a single USB camera.

This script mirrors `record_episodes_custom_alicia.py` but replaces the
RealSense RGB-D pipeline with the USB camera capture utilities from
`record_sim_teleop_TCP_EE_USB_cam.py`.
"""
import argparse
import os
from time import sleep, time

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import alicia_d_sdk

from config.config import TASK_CONFIG, ROBOT_PORTS


def initialize_usb_camera(device_id, width, height):
    """Open a persistent USB camera handle."""
    print(f"Initializing USB camera on /dev/video{device_id}...")
    camera = cv2.VideoCapture(device_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not camera.isOpened():
        print(f"Failed to open USB camera /dev/video{device_id}")
        return None

    return camera


def capture_usb_frame(camera, target_width, target_height):
    """Read, convert, and resize a frame from an open USB camera handle."""
    try:
        if camera is None or not camera.isOpened():
            print("Warning: USB camera is not available")
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        ret, frame = camera.read()
        if not ret or frame is None:
            print("Warning: Failed to capture USB camera frame")
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image.shape[0] != target_height or image.shape[1] != target_width:
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return image

    except Exception as exc:  # pragma: no cover - defensive hardware handling
        print(f"Error capturing USB camera image: {exc}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)


def read_robot_state(robot):
    """Read robot joint state, velocities, and end-effector pose via Alicia SDK."""
    try:
        joint_gripper_state = robot.get_robot_state("joint_gripper")
        if joint_gripper_state is None:
            print("Failed to get joint_gripper state")
            return np.zeros(7), np.zeros(7), np.zeros(3), np.zeros(4)

        joint_angles = joint_gripper_state.angles
        gripper_pos = joint_gripper_state.gripper

        pose_data = robot.get_pose()
        if pose_data is None:
            print("Failed to get pose data")
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
        else:
            ee_pos = np.array(pose_data['position'])
            ee_quat = np.array(pose_data['quaternion_xyzw'])

        gripper_normalized = 1 - gripper_pos / 1000.0
        qpos = joint_angles + [gripper_normalized]
        qvel = [0.0] * 7

        return np.array(qpos), np.array(qvel), ee_pos, ee_quat
    except Exception as exc:  # pragma: no cover - defensive hardware handling
        print(f"Error reading robot state: {exc}")
        return np.zeros(7), np.zeros(7), np.zeros(3), np.zeros(4)


def set_robot_position(robot, target_pos):
    """Command robot joints (6 DOF) plus gripper via Alicia SDK."""
    try:
        joint_targets = target_pos[:6].tolist()
        gripper_target = target_pos[6]

        gripper_target_raw = int(gripper_target * 1000)
        gripper_target_raw = max(0, min(1000, gripper_target_raw))

        success = robot.set_robot_state(
            target_joints=joint_targets,
            gripper_value=gripper_target_raw,
            joint_format='rad',
            speed_deg_s=30,
            wait_for_completion=False,
        )

        if not success:
            print("Failed to set robot position")
    except Exception as exc:  # pragma: no cover - defensive hardware handling
        print(f"Error setting robot position: {exc}")


def get_next_episode_index(data_dir):
    """Return next available episode index inside data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    episode_files = [f for f in os.listdir(data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    if not episode_files:
        return 0

    indices = []
    for filename in episode_files:
        try:
            indices.append(int(filename.split('_')[1].split('.')[0]))
        except (IndexError, ValueError):
            continue

    return max(indices) + 1 if indices else 0


def build_arg_parser(default_cfg):
    parser = argparse.ArgumentParser(description="Record teleop episodes with USB camera")
    parser.add_argument('--task', type=str, default='task1a', help='Task name / dataset sub-folder')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to record')
    parser.add_argument('--episode_len', type=int, default=default_cfg['episode_len'], help='Episode length in timesteps')
    parser.add_argument('--dataset_dir', type=str, default=default_cfg['dataset_dir'], help='Root directory for saved datasets')
    parser.add_argument('--camera_id', type=int, default=default_cfg.get('camera_port', 0), help='USB camera device id (/dev/videoX)')
    parser.add_argument('--cam_width', type=int, default=default_cfg['cam_width'], help='Camera capture width')
    parser.add_argument('--cam_height', type=int, default=default_cfg['cam_height'], help='Camera capture height')
    parser.add_argument('--sync_steps', type=int, default=200, help='Steps to sync follower to leader before recording')
    return parser


def save_episode(data_dict, data_dir, episode_idx, cfg):
    """Persist a single episode to HDF5."""
    dataset_path = os.path.join(data_dir, f'episode_{episode_idx}.hdf5')
    max_timesteps = len(data_dict['/observations/qpos'])

    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')

        height = cfg['cam_height']
        width = cfg['cam_width']
        image.create_dataset('front', (max_timesteps, height, width, 3), dtype='uint8', chunks=(1, height, width, 3))

        qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        ee_pos = obs.create_dataset('ee_pos', (max_timesteps, 3))
        ee_quat = obs.create_dataset('ee_quat', (max_timesteps, 4))
        action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))

        qpos[...] = data_dict['/observations/qpos']
        qvel[...] = data_dict['/observations/qvel']
        ee_pos[...] = data_dict['/observations/ee_pos']
        ee_quat[...] = data_dict['/observations/ee_quat']
        action[...] = data_dict['/action']
        image['front'][...] = data_dict['/observations/images/front']

    return dataset_path


def main():
    cfg = dict(TASK_CONFIG)
    parser = build_arg_parser(cfg)
    args = parser.parse_args()

    task = args.task
    num_episodes = args.num_episodes
    episode_len = args.episode_len
    dataset_root = args.dataset_dir
    camera_id = args.camera_id
    cam_width = args.cam_width
    cam_height = args.cam_height
    sync_steps = args.sync_steps

    data_dir = os.path.join(dataset_root, task)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Dataset directory: {data_dir}")

    camera = initialize_usb_camera(camera_id, cam_width, cam_height)
    if camera is None:
        raise RuntimeError("Unable to open USB camera. Aborting.")

    follower = None
    leader = None
    try:
        print("Initializing robots...")
        follower = alicia_d_sdk.create_robot(port=ROBOT_PORTS['follower'], gripper_type="50mm")
        print("✓ Follower robot connected")
        leader = alicia_d_sdk.create_robot(port=ROBOT_PORTS['leader'], gripper_type="50mm")
        print("✓ Leader robot connected")
    except Exception as exc:
        if camera is not None:
            camera.release()
        raise RuntimeError(f"Failed to initialize robots: {exc}")

    try:
        for episode_idx in range(num_episodes):
            print(f"\nStarting episode {episode_idx + 1}/{num_episodes}")
            print("Syncing follower to leader position...")
            for _ in range(sync_steps):
                leader_qpos, _, _, _ = read_robot_state(leader)
                set_robot_position(follower, leader_qpos)
                sleep(0.01)

            print("\n" + "=" * 60)
            print("🔴 RECORDING START!")
            print("=" * 60)

            obs_replay = []
            action_replay = []

            for _ in tqdm(range(episode_len), desc=f"Episode {episode_idx + 1}"):
                qpos, qvel, ee_pos, ee_quat = read_robot_state(follower)
                usb_image = capture_usb_frame(camera, cam_width, cam_height)

                obs = {
                    'qpos': qpos,
                    'qvel': qvel,
                    'ee_pos': ee_pos,
                    'ee_quat': ee_quat,
                    'images': {'front': usb_image},
                }

                action, _, _, _ = read_robot_state(leader)
                set_robot_position(follower, action)

                obs_replay.append(obs)
                action_replay.append(action)

            print("\n" + "🛑" * 20)
            print("🛑 RECORDING STOPPED 🛑")
            print("🛑" * 20 + "\n")

            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/observations/ee_pos': [],
                '/observations/ee_quat': [],
                '/observations/images/front': [],
                '/action': [],
            }

            for obs, action in zip(obs_replay, action_replay):
                data_dict['/observations/qpos'].append(obs['qpos'])
                data_dict['/observations/qvel'].append(obs['qvel'])
                data_dict['/observations/ee_pos'].append(obs['ee_pos'])
                data_dict['/observations/ee_quat'].append(obs['ee_quat'])
                data_dict['/observations/images/front'].append(obs['images']['front'])
                data_dict['/action'].append(action)

            for key in data_dict:
                data_dict[key] = np.asarray(data_dict[key])

            next_idx = get_next_episode_index(data_dir)
            start_time = time()
            output_path = save_episode(data_dict, data_dir, next_idx, {
                'cam_height': cam_height,
                'cam_width': cam_width,
                'state_dim': cfg['state_dim'],
                'action_dim': cfg['action_dim'],
            })
            duration = time() - start_time
            print(f"Episode saved to {output_path} in {duration:.1f}s")

    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    finally:
        print("Cleaning up resources...")
        if leader is not None:
            try:
                leader.disconnect()
                print("✓ Leader robot disconnected")
            except Exception as exc:
                print(f"Leader disconnect warning: {exc}")
        if follower is not None:
            try:
                follower.disconnect()
                print("✓ Follower robot disconnected")
            except Exception as exc:
                print(f"Follower disconnect warning: {exc}")
        if camera is not None:
            camera.release()
            print("✓ USB camera released")


if __name__ == '__main__':
    main()
