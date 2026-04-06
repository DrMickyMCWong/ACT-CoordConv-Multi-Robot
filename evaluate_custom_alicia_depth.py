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
import pyrealsense2 as rs  # RealSense SDK for L515 camera
from training.utils import *


# Control loop parameters (same as replay script)
TOR_Q = 0.005          # Joint position tolerance (radians) - consider "reached" if within this
GRIP_TOL = 30         # Gripper tolerance (0-1000 scale) - consider "reached" if within this
MAX_WAIT = 0.35       # Maximum wait time per step (seconds) before moving to next command
GRIP_MIN = 0          # Fully open (robot's 0-1000 scale)
GRIP_MAX_SAFE = 950   # Safe maximum closure (slightly less than 1000 to avoid hard-stop stall)


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1c')
parser.add_argument('--delay', type=float, default=0.05,
                    help='Delay between commands in seconds')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']


# Cross-platform audio feedback
def play_sound(type="start"):
    if type == "start":
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("🚀 EVALUATION STARTED")
        print("="*50 + "\n")
    else:
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("✅ EVALUATION STOPPED")
        print("="*50 + "\n")


# Cross-platform audio feedback
def play_sound(type="start"):
    if type == "start":
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("🚀 EVALUATION STARTED")
        print("="*50 + "\n")
    else:
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("✅ EVALUATION STOPPED")
        print("="*50 + "\n")


def read_robot_state(robot, timeout=2.0):
    """Read robot joint angles and gripper state using Alicia SDK"""
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Robot state reading timed out")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            # Get joint angles and gripper using unified API
            joint_gripper_state = robot.get_robot_state("joint_gripper")  # Returns JointState object
            
            # Clear the alarm
            signal.alarm(0)
            
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
            
        except TimeoutError:
            print("Warning: Robot state reading timed out")
            return np.zeros(7), np.zeros(7)
        finally:
            # Make sure to clear the alarm
            signal.alarm(0)
    
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


def capture_image_rgbd(pipeline, target_width=640, target_height=480):
    """Capture both RGB and depth images from RealSense L515 camera"""
    try:
        # Wait for frames with timeout
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        
        # Align depth to color frame
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Warning: Failed to capture camera frames")
            # Return black frames if capture fails
            rgb_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            depth_image = np.zeros((target_height, target_width), dtype=np.uint16)
            return rgb_image, depth_image
        
        # Convert color frame (RealSense returns BGR format)
        rgb_frame = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        # Get depth frame data (uint16 in millimeters)
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Resize both to target dimensions
        rgb_image = cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        depth_image = cv2.resize(depth_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return rgb_image, depth_image
    
    except Exception as e:
        print(f"Error capturing RGBD images: {e}")
        rgb_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        depth_image = np.zeros((target_height, target_width), dtype=np.uint16)
        return rgb_image, depth_image


def capture_image_usb(device_id=0, target_width=640, target_height=480):
    """Capture image from USB camera (e.g. side/hand camera) using OpenCV VideoCapture.

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


def get_image_eval(images, camera_names, device='cpu'):
    """
    Process images for evaluation - matches EpisodicDataset.__getitem__() exactly
    This function replicates the same depth processing as training
    """
    curr_images = []
    for cam_name in camera_names:
        image = images[cam_name]
        
        # Handle depth images - same processing as training/utils.py EpisodicDataset
        if cam_name.endswith('_depth'):
            # Depth image: (H, W) -> (H, W, 3)
            # Convert from uint16 millimeters to float32 meters
            image = image.astype(np.float32) / 1000.0  # Convert mm to meters
            
            # Clip depth values to manipulation range (0.2m to 0.8m based on data analysis)
            image = np.clip(image, 0.2, 0.8)
            
            # Normalize depth to 0-255 range - same value always means same real distance
            image = ((image - 0.2) / (0.8 - 0.2) * 255.0).astype(np.uint8)
            
            # Add channel dimension and replicate to 3 channels: (H, W) -> (H, W, 3)
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
            image = np.repeat(image, 3, axis=-1)  # Replicate to 3 channels
        
        # Rearrange from 'h w c' to 'c h w' (same as training utils)
        curr_image = rearrange(image, 'h w c -> c h w')
        curr_images.append(curr_image)
    
    # Stack cameras and add batch dimension (same as training utils)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def capture_image(pipeline):
    """Capture and resize camera frame using RealSense SDK (RGB only - legacy function)"""
    rgb_image, _ = capture_image_rgbd(pipeline, cfg['cam_width'], cfg['cam_height'])
    return rgb_image


if __name__ == "__main__":
    # init camera using RealSense SDK
    print("Initializing L515 camera with RealSense SDK...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream - use fixed 640x480 like working record script
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Enable depth stream - let it use native resolution, then align to color
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        print("✓ L515 camera started successfully")
        
        # Create align object to align depth frames to color frames
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Get device info
        device_info = profile.get_device()
        print(f"Device: {device_info.get_info(rs.camera_info.name)}")
        print(f"Serial: {device_info.get_info(rs.camera_info.serial_number)}")
        
        # Allow camera to settle - match working script
        print("Allowing camera to settle...")
        for i in range(30):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
        print("✓ Camera ready for capture")
            
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
    
    # Print camera configuration for verification
    print(f"📷 Camera configuration:")
    print(f"   Cameras: {cfg['camera_names']}")
    print(f"   Target resolution: {cfg['cam_width']}x{cfg['cam_height']}")
    
    stats_path = os.path.join(train_cfg['checkpoint_dir'], task, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # Prepare for evaluation
    print("\nTesting camera capture...")
    test_images = capture_all_cameras(pipeline)
    for cam_name, img in test_images.items():
        if cam_name == 'front_depth':
            print(f"   {cam_name}: {img.shape}, dtype: {img.dtype}")
        else:
            print(f"   {cam_name}: {img.shape}, dtype: {img.dtype}")
    
    print("\nPreparing for evaluation in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    # Start evaluation with sound cue
    play_sound("start")

    # Get initial observation using new API
    print("Getting initial observation...")
    try:
        qpos, qvel = read_robot_state(follower)
        print(f"  Initial robot state: qpos shape={np.array(qpos).shape}, qvel shape={np.array(qvel).shape}")
        
        raw_images = capture_all_cameras(pipeline)
        print(f"  Captured images: {[f'{k}: {v.shape}' for k, v in raw_images.items()]}")
        
        processed_images = get_image_eval(raw_images, cfg['camera_names'], device=device)
        print(f"  Processed images shape: {processed_images.shape}")
        
        obs = {
            'qpos': qpos,  # 7-DOF array
            'qvel': qvel,  # 7-DOF array  
            'images': processed_images,  # Processed for policy
            'raw_images': raw_images     # Raw for storage
        }
        print("✓ Initial observation ready")
        
    except Exception as e:
        print(f"Error getting initial observation: {e}")
        print("Exiting due to initialization failure...")
        follower.disconnect()
        pipeline.stop()
        exit(1)
    

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
                curr_image = obs['images']  # Already processed by get_image_eval()

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
                raw_images = capture_all_cameras(pipeline)
                processed_images = get_image_eval(raw_images, cfg['camera_names'], device=device)
                obs = {
                    'qpos': qpos,  # 7-DOF array
                    'qvel': qvel,  # 7-DOF array
                    'images': processed_images,  # Processed for policy
                    'raw_images': raw_images     # Raw for storage
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)

        # End evaluation with sound cue
        play_sound("stop")

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
            # Store RAW images for HDF5 (not processed tensors)
            for cam_name in cfg['camera_names']:
                raw_image = o['raw_images'][cam_name]
                
                # Handle depth images - save as uint16 like training data
                if cam_name.endswith('_depth'):
                    # Save depth as uint16 in millimeters (raw format)
                    data_dict[f'/observations/images/{cam_name}'].append(raw_image)
                else:
                    # RGB image - store as-is
                    data_dict[f'/observations/images/{cam_name}'].append(raw_image)

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
    try:
        pipeline.stop()
        print("✓ Camera stopped")
    except Exception as e:
        print(f"Note: Pipeline stop error (normal if already stopped): {e}")