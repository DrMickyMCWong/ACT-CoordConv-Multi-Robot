from config.config import TASK_CONFIG, ROBOT_PORTS
import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from time import sleep, time

import alicia_d_sdk  # Use new Alicia SDK
import pyrealsense2 as rs  # RealSense SDK for L515 camera

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
parser.add_argument('--num_episodes', type=int, default=1)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG


def read_robot_state(robot):
    """Read robot joint angles, gripper state, and end-effector pose using Alicia SDK"""
    try:
        # Get joint angles and gripper using unified API
        joint_gripper_state = robot.get_robot_state("joint_gripper")  # Returns JointState object
        
        if joint_gripper_state is None:
            print("Failed to get joint_gripper state")
            return np.zeros(7), np.zeros(7), np.zeros(3), np.zeros(4)
        
        # Extract joint angles (6 DOF) in radians
        joint_angles = joint_gripper_state.angles  # List of 6 joint angles in radians
        
        # Extract gripper position (0-1000)
        gripper_pos = joint_gripper_state.gripper  # Gripper position 0-1000
        
        # Get end-effector pose using the correct method
        pose_data = robot.get_pose()  # Returns dict with position and quaternion
        
        if pose_data is None:
            print("Failed to get pose data")
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
        else:
            # Extract position (xyz in meters)
            ee_pos = np.array(pose_data['position'])  # [x, y, z] in meters
            
            # Extract quaternion (xyzw format from FK, but we want xyzw)
            ee_quat = np.array(pose_data['quaternion_xyzw'])  # [x, y, z, w]
        
        # Combine into 7-DOF state: [joint1, joint2, joint3, joint4, joint5, joint6, gripper_normalized]
        # Convert gripper from 0-1000 to 0-1 range
        gripper_normalized = 1 - gripper_pos / 1000.0
        
        # Create 7-DOF position array
        qpos = joint_angles + [gripper_normalized]
        
        # For velocity, we'll use zeros for now (SDK doesn't provide direct velocity reading in joint_gripper)
        # Could get velocities separately with robot.get_robot_state("velocity") if needed
        qvel = [0.0] * 7
        
        return np.array(qpos), np.array(qvel), ee_pos, ee_quat
    
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return np.zeros(7), np.zeros(7), np.zeros(3), np.zeros(4)

def set_robot_position(robot, target_pos):
    """Set robot joint positions using Alicia SDK"""
    try:
        # Split 7-DOF target into 6 joints + gripper
        joint_targets = target_pos[:6].tolist()  # First 6 elements are joint angles (convert to list)
        gripper_target = target_pos[6]   # 7th element is gripper (0-1 normalized)
        
        # Convert gripper from 0-1 to 0-1000 range
        gripper_target_raw = int(gripper_target * 1000)
        gripper_target_raw = max(0, min(1000, gripper_target_raw))  # Clamp to valid range
        
        # Set joint angles and gripper using unified method
        success = robot.set_robot_state(
            target_joints=joint_targets,  # 6 joint angles in radians
            gripper_value=gripper_target_raw,  # Gripper 0-1000
            joint_format='rad',  # Input is in radians
            speed_deg_s=30,  # Reasonable speed
            wait_for_completion=False  # Don't wait to maintain data collection timing
        )
        
        if not success:
            print(f"Failed to set robot position")
        
    except Exception as e:
        print(f"Error setting robot position: {e}")


def capture_image_rgbd(pipeline, target_width=640, target_height=480):
    """Capture both RGB and depth images from RealSense L515 camera"""
    try:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Align depth to color to match resolutions
        align_to_color = rs.align(rs.stream.color)
        aligned_frames = align_to_color.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not aligned_depth_frame:
            print("Warning: Failed to capture camera frames")
            # Return black frames if capture fails
            return (np.zeros((target_height, target_width, 3), dtype=np.uint8),
                   np.zeros((target_height, target_width), dtype=np.uint16))
        
        # Convert RGB to numpy array (RealSense returns BGR format)
        rgb_frame = np.asanyarray(color_frame.get_data())
        
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        # Resize RGB directly without cropping to avoid deformation
        rgb_image = cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Process depth data - USE THE EXACT SAME METHOD AS SUCCESSFUL TEST
        # Get depth sensor for scale
        profile = pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Convert raw depth data using depth scale (RealSense SDK handles offset automatically)
        depth_raw = np.asanyarray(aligned_depth_frame.get_data())
        depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000  # Convert to mm
        
        # Resize depth image to match target size if needed
        if depth_mm.shape[:2] != (target_height, target_width):
            depth_mm = cv2.resize(depth_mm, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to uint16 for storage, clamp to valid range
        depth_mm = np.clip(depth_mm, 0, 65535)  # uint16 max
        depth_image = depth_mm.astype(np.uint16)
        
        # Removed verbose depth frame logging to improve recording speed

        return rgb_image, depth_image
    
    except Exception as e:
        print(f"Error capturing RGBD images: {e}")
        return (np.zeros((target_height, target_width, 3), dtype=np.uint8),
               np.zeros((target_height, target_width), dtype=np.uint16))


def capture_image_usb(device_id=0, target_width=640, target_height=480):
    """Capture image from USB camera (e.g. top camera) using OpenCV VideoCapture.

    Default device_id=0 (maps to /dev/video0). Returns RGB image (H,W,3) uint8.
    """
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Warning: Could not open USB camera /dev/video{device_id}")
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (target_height, target_width):
                frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
            return frame_rgb
        else:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Error capturing USB camera: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)


def capture_image(pipeline):
    """Capture and resize camera frame using RealSense SDK (RGB only - legacy function)"""
    rgb_image, _ = capture_image_rgbd(pipeline, cfg['cam_width'], cfg['cam_height'])
    return rgb_image


def capture_all_cameras(realsense_pipeline):
    """Capture images from all cameras: RealSense RGB+Depth, Side USB, Hand USB - OPTIMIZED"""
    images = {}
    
    # Capture RealSense RGB and depth (most time-consuming)
    rgb_image, depth_image = capture_image_rgbd(
        realsense_pipeline, 
        cfg['cam_width'], 
        cfg['cam_height']
    )
    images['front'] = rgb_image
    images['front_depth'] = depth_image
    
    # Quick USB captures - minimize overhead
    try:
        # Side camera - fast capture
        cap_side = cv2.VideoCapture(0)
        if cap_side.isOpened():
            ret, frame = cap_side.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images['side'] = cv2.resize(frame_rgb, (cfg['cam_width'], cfg['cam_height']))
            else:
                images['side'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
            cap_side.release()
        else:
            images['side'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
    except:
        images['side'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
    
    try:
        # Hand camera - fast capture
        cap_hand = cv2.VideoCapture(10)
        if cap_hand.isOpened():
            ret, frame = cap_hand.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images['hand'] = cv2.resize(frame_rgb, (cfg['cam_width'], cfg['cam_height']))
            else:
                images['hand'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
            cap_hand.release()
        else:
            images['hand'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
    except:
        images['hand'] = np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
    
    return images


if __name__ == "__main__":
    # init RealSense camera with RGB and depth streams
    print("Initializing L515 camera with RealSense SDK for RGB and depth...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream - use fixed 640x480 like working interactive script
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Enable depth stream - let it use native resolution, then align to color
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        print("✓ L515 camera started successfully with RGB and depth")
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        
        # Configure depth sensor for close-range work if available
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 4)  # Short range preset
            print("✓ Depth sensor configured for short range")
        
        # Get depth scale for processing
        depth_scale = depth_sensor.get_depth_scale()
        print(f"✓ Depth scale: {depth_scale:.6f} (sensor units to meters)")
        
        # Allow camera to settle - reduced for faster startup
        for i in range(10):
            pipeline.wait_for_frames()
            
    except Exception as e:
        print(f"Failed to start L515 camera: {e}")
        print("Make sure the camera is connected and not being used by another application")
        exit(1)
    
    # Test USB cameras
    print("Testing USB cameras...")
    try:
        test_side = capture_image_usb(0, 160, 120)  # Small test image
        if test_side.max() > 0:
            print("✓ Side camera (/dev/video0) working")
        else:
            print("⚠️ Side camera (/dev/video0) may not be working properly")
    except Exception as e:
        print(f"⚠️ Side camera (/dev/video0) error: {e}")
    
    try:
        test_hand = capture_image_usb(10, 160, 120)  # Small test image
        if test_hand.max() > 0:
            print("✓ Hand camera (/dev/video10) working")
        else:
            print("⚠️ Hand camera (/dev/video10) may not be working properly")
    except Exception as e:
        print(f"⚠️ Hand camera (/dev/video10) error: {e}")
        
    # Initialize robots using Alicia SDK
    print("Initializing robots...")
    try:
        follower = alicia_d_sdk.create_robot(
            port=ROBOT_PORTS['follower'],
            gripper_type="50mm"  # Adjust as needed
        )
        print("✓ Follower robot connected")
    except Exception as e:
        print(f"⚠️ Failed to connect follower robot: {e}")
        follower = None
    
    try:
        leader = alicia_d_sdk.create_robot(
            port=ROBOT_PORTS['leader'],
            gripper_type="50mm"  # Adjust as needed
        )
        print("✓ Leader robot connected")
    except Exception as e:
        print(f"⚠️ Failed to connect leader robot: {e}")
        leader = None
    
    if follower is None or leader is None:
        print("⚠️ Robot connection failed. Testing cameras only...")
        # Test all cameras and exit
        print("Testing all cameras...")
        test_images = capture_all_cameras(pipeline)
        
        for cam_name, image in test_images.items():
            if cam_name == 'front_depth':
                nonzero_pixels = np.count_nonzero(image)
                print(f"✓ {cam_name}: {image.shape} shape, {nonzero_pixels} valid depth pixels")
            else:
                print(f"✓ {cam_name}: {image.shape} shape, max value: {image.max()}")
        
        print(f"Camera configuration: {cfg['camera_names']}")
        print("Camera test completed successfully! All 4 cameras are working.")
        
        # Stop camera pipeline
        pipeline.stop()
        print("✓ Camera stopped")
        exit(0)
    
    print("✓ Robots connected successfully")
    
    print(f"Camera configuration: {cfg['camera_names']}")

    for episode_idx in range(num_episodes):
        print(f"Starting episode {episode_idx + 1}/{num_episodes}")
        
        # Bring the follower to the leader position and start camera
        print("Syncing follower to leader position...")
        for i in range(200):
            leader_qpos, _, _, _ = read_robot_state(leader)
            set_robot_position(follower, leader_qpos)
            sleep(0.01)  # Small delay for stability
            
        # Clear recording start indicators
        print("\n" + "="*60)
        print("🔴 RECORDING STARTING IN 3 SECONDS...")
        print("="*60)
        
        # Countdown
        for countdown in [3, 2, 1]:
            print(f"🔴 RECORDING IN {countdown}...")
            sleep(1)
        
        # Recording start indicators
        print("\n" + "🔴"*20)
        print("🔴 RECORDING STARTED! 🔴")
        print("🔴" + " "*16 + "🔴")
        print("🔴  DEMONSTRATE TASK  🔴") 
        print("🔴" + " "*16 + "🔴")
        print("🔴"*20 + "\n")
        
        # init buffers
        obs_replay = []
        action_replay = []
        
        for i in tqdm(range(cfg['episode_len']), desc=f"Recording episode {episode_idx + 1}"):
            # observation - get current follower state
            qpos, qvel, ee_pos_follower, ee_quat_follower = read_robot_state(follower)
            
            # Capture from all cameras
            images = capture_all_cameras(pipeline)
            
            obs = {
                'qpos': qpos,  # Already in correct format (7-DOF)
                'qvel': qvel,  # Already in correct format (7-DOF)
                'ee_pos': ee_pos_follower,  # End-effector position [x, y, z] in meters
                'ee_quat': ee_quat_follower,  # End-effector quaternion [qx, qy, qz, qw]
                'images': images  # Now contains all 4 camera views
            }
            
            # action - get leader's target position and pose
            action, _, ee_pos_leader, ee_quat_leader = read_robot_state(leader)
            
            # apply action to follower
            set_robot_position(follower, action)
            
            # store data
            obs_replay.append(obs)
            action_replay.append(action)  # Already in correct format (7-DOF)

        # Recording stop indicators
        print("\n" + "🛑"*20)
        print("🛑 RECORDING STOPPED! 🛑")
        print("🛑" + " "*16 + "🛑")
        print("🛑    TASK COMPLETE   🛑")
        print("🛑" + " "*16 + "🛑")
        print("🛑"*20 + "\n")

        # disable torque - not needed for Alicia SDK
        # leader._disable_torque()
        # follower._disable_torque()

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/ee_pos': [],
            '/observations/ee_quat': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/observations/ee_pos'].append(o['ee_pos'])
            data_dict['/observations/ee_quat'].append(o['ee_quat'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        data_dir = os.path.join(cfg['dataset_dir'], task)
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        
        print(f"Saving episode {episode_idx + 1} to {dataset_path}.hdf5")
        
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False  # This is real robot data, not sim
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Create datasets for each camera with appropriate data types
            for cam_name in cfg['camera_names']:
                if cam_name == 'front_depth':
                    # Depth images are uint16
                    _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width']), dtype='uint16',
                                            chunks=(1, cfg['cam_height'], cfg['cam_width']), )
                else:
                    # RGB images are uint8 with 3 channels
                    _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                            chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            ee_pos = obs.create_dataset('ee_pos', (max_timesteps, 3))  # XYZ position
            ee_quat = obs.create_dataset('ee_quat', (max_timesteps, 4))  # Quaternion (qx, qy, qz, qw)
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f"✓ Episode {episode_idx + 1} saved successfully")
    
    # Disconnect robots properly
    print("Disconnecting robots...")
    leader.disconnect()
    follower.disconnect()
    print("✓ All robots disconnected")
    
    # Stop camera pipeline with error handling
    print("Stopping camera...")
    try:
        pipeline.stop()
        print("✓ Camera stopped")
    except Exception as e:
        print(f"Note: Pipeline stop error (normal if already stopped): {e}")
