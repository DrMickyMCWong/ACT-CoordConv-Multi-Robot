from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import warnings
# Suppress verbose PyTorch warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import cv2
import torch
import pickle
import argparse
import time  # Import time module, not function
import h5py  # Add this import
import numpy as np
from tqdm import tqdm  # Add for progress display

import alicia_d_sdk  # Use new Alicia SDK instead of old Robot class
from training.utils import *


# Control loop parameters (same as replay script)
TOR_Q = 0.005          # Joint position tolerance (radians) - consider "reached" if within this
GRIP_TOL = 30         # Gripper tolerance (0-1000 scale) - consider "reached" if within this
MAX_WAIT = 0.35       # Maximum wait time per step (seconds) before moving to next command
GRIP_MIN = 0          # Fully open (robot's 0-1000 scale)
GRIP_MAX_SAFE = 950   # Safe maximum closure (slightly less than 1000 to avoid hard-stop stall)


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1a')
parser.add_argument('--delay', type=float, default=0.05,
                    help='Delay between commands in seconds')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']



def read_robot_state(robot):
    """Read robot joint angles and gripper state using Alicia SDK"""
    try:
        # Get joint angles and gripper using unified API
        joint_gripper_state = robot.get_robot_state("joint_gripper")  # Returns JointState object
        
        if joint_gripper_state is None:
            print("Failed to get joint_gripper state")
            return np.zeros(7), np.zeros(7)
        
        # Extract joint angles (6 DOF) in radians
        joint_angles = joint_gripper_state.angles  # List of 6 joint angles in radians
        
        # Extract gripper position (0-1000)
        gripper_pos = joint_gripper_state.gripper  # Gripper position 0-1000
        
        # Combine into 7-DOF state: [joint1, joint2, joint3, joint4, joint5, joint6, gripper_normalized]
        # Convert gripper from 0-1000 to 0-1 range
        gripper_normalized = 1 - gripper_pos / 1000.0
        
        # Create 7-DOF position array
        qpos = joint_angles + [gripper_normalized]
        
        # For velocity, we'll use zeros for now (SDK doesn't provide direct velocity reading in joint_gripper)
        # Could get velocities separately with robot.get_robot_state("velocity") if needed
        qvel = [0.0] * 7
        
        return np.array(qpos), np.array(qvel)
    
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return np.zeros(7), np.zeros(7)

def step_with_tolerance(robot, target_pos, max_wait=MAX_WAIT):
    """
    Send command and wait until robot is "close enough" to target or timeout expires.
    
    This approach:
    - Sends command with wait_for_completion=False (non-blocking)
    - Polls current position and checks if within tolerance
    - Moves to next command once "close enough" or timeout expires
    - Prevents both stalling (timeout) and skipping (tolerance check)
    
    Args:
        robot: Robot SDK instance
        target_pos: Target 7-DOF position [6 joints + gripper normalized 0-1]
        max_wait: Maximum time to wait for convergence (seconds)
    
    Returns:
        bool: True if target reached within tolerance, False if timeout
    """
    try:
        # Split 7-DOF target into 6 joints + gripper
        if hasattr(target_pos, 'tolist'):
            joint_targets = target_pos[:6].tolist()
        else:
            joint_targets = target_pos[:6]
        gripper_target = target_pos[6]  # 0-1 normalized
        
        # Convert gripper: policy 0=closed, 1=open → robot 1000=closed, 0=open
        gripper_target_raw = int((1 - gripper_target) * 1000)
        # Clamp gripper to safe range
        gripper_target_raw = int(np.clip(gripper_target_raw, GRIP_MIN, GRIP_MAX_SAFE))
        
        # Send command without blocking
        robot.set_robot_state(
            target_joints=joint_targets,
            gripper_value=gripper_target_raw,
            joint_format='rad',
            speed_deg_s=10,  # Fixed speed for evaluation
            wait_for_completion=False,  # Non-blocking
            gripper_speed_deg_s=100
        )
        
        # Poll until "close enough" or timeout
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Read current state
            joint_gripper_state = robot.get_robot_state("joint_gripper")
            if joint_gripper_state is None:
                time.sleep(0.01)
                continue
            
            current_joints = np.array(joint_gripper_state.angles)  # 6 joints in radians
            current_gripper = joint_gripper_state.gripper  # 0-1000 scale
            
            # Check joint convergence: max absolute error across all joints
            joint_error = np.max(np.abs(current_joints - np.array(joint_targets)))
            
            # Check gripper convergence
            gripper_error = abs(current_gripper - gripper_target_raw)
            
            # Consider "reached" if both joints and gripper are within tolerance
            if joint_error < TOR_Q and gripper_error < GRIP_TOL:
                return True  # Success - reached target
            
            time.sleep(0.01)  # Poll every 10ms
        
        # Timeout - didn't reach target, but move on to prevent stalling
        return False
        
    except Exception as e:
        print(f"Error in step_with_tolerance: {e}")
        return False

def set_robot_position(robot, target_pos):
    """
    Set robot position using step-with-tolerance approach.
    Wrapper function for compatibility with existing evaluation code.
    """
    return step_with_tolerance(robot, target_pos, max_wait=MAX_WAIT)


def capture_image(camera):
    """Capture and resize camera frame using USB camera (/dev/video0)."""
    try:
        if camera is None or not camera.isOpened():
            print("Warning: USB camera not available")
            return np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)

        ret, frame = camera.read()
        if not ret or frame is None:
            print("Warning: Failed to capture USB camera frame")
            return np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image.shape[0] != cfg['cam_height'] or image.shape[1] != cfg['cam_width']:
            image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)

        return image

    except Exception as e:
        print(f"Error capturing USB camera image: {e}")
        return np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)


if __name__ == "__main__":
    # init USB camera (/dev/video0)
    print("Initializing USB camera (/dev/video0)...")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['cam_width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['cam_height'])

    if not camera.isOpened():
        print("Failed to open USB camera. Make sure /dev/video0 is available.")
        exit(1)
        
    # Initialize robot using Alicia SDK
    print("Initializing follower robot...")
    follower = alicia_d_sdk.create_robot(
        port=ROBOT_PORTS['follower'],
        gripper_type="50mm"  # Adjust as needed
    )
    
    print("✓ Robot connected successfully")
    
    # Wait for stable communication
    print("Waiting for stable communication...")
    time.sleep(2)

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], task, train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], task, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # Prepare for evaluation
    print("\nPreparing for evaluation in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    # Start evaluation message (replaces audio cue)
    print("\n" + "="*50)
    print("🚀 EVALUATION STARTED")
    print("="*50 + "\n")

    # bring the follower to ready position
    for i in range(90):
        qpos, qvel = read_robot_state(follower)
        _ = capture_image(camera)
    
    # Get initial observation using new API
    qpos, qvel = read_robot_state(follower)
    obs = {
        'qpos': qpos,  # 7-DOF array
        'qvel': qvel,  # 7-DOF array  
        'images': {cn: capture_image(camera) for cn in cfg['camera_names']}
    }
    

    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            
            # Add progress bar for better feedback
            for t in tqdm(range(cfg['episode_len'])):
                qpos_numpy = np.array(obs['qpos'])
                qpos = (qpos_numpy - stats['qpos_mean']) / stats['qpos_std']  # Apply normalization
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image) # Shape: [1, 100, 6]
                # Complex temporal aggregation logic
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1) # mask
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    # Pick one action from the sequence
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = raw_action * stats['action_std'] + stats['action_mean']  # Apply denormalization
                
                # Convert to standard Python float list for JSON serialization
                action = action.astype(float).tolist()
                
                ### take action using new API
                set_robot_position(follower, action)

                ### Optional: Add small delay between commands (same as replay script)
                # The tolerance check already provides natural pacing
                if args.delay > 0:
                    time.sleep(args.delay)

                ### update obs using new API
                qpos, qvel = read_robot_state(follower)
                obs = {
                    'qpos': qpos,  # 7-DOF array
                    'qvel': qvel,  # 7-DOF array
                    'images': {cn: capture_image(camera) for cn in cfg['camera_names']}
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)

        # End evaluation message (replaces audio cue)
        print("\n" + "="*50)
        print("✅ EVALUATION STOPPED")
        print("="*50 + "\n")

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        # Use time.time() instead of time()
        t0 = time.time()
        max_timesteps = len(data_dict['/observations/qpos'])
        
        # create data dir if it doesn't exist
        eval_dir = os.path.join(cfg['dataset_dir'], "evaluation", task)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        
        # Generate a filename with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(eval_dir, f'eval_result_{task}_{timestamp}.hdf5')

        print(f"Will save evaluation results to: {filepath}")

        with h5py.File(filepath, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
            
            print(f"✅ Evaluation completed and data saved to {filepath}")
    
    # Disconnect robot properly
    print("Disconnecting robot...")
    follower.disconnect()
    print("✓ Robot disconnected")
    
    # Stop camera pipeline with error handling
    print("Stopping camera...")
    if camera is not None and camera.isOpened():
        camera.release()
    print("✓ Camera stopped")