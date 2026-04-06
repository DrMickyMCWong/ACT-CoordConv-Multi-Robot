from config.config_two_cam import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import cv2
import torch
import pickle
import argparse
import time  # Import time module, not function
import winsound  # For Windows sound
import h5py  # Add this import
import numpy as np
from tqdm import tqdm  # Add for progress display

from robot import Robot
from training.utils import *


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task5c')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

def capture_image(cam, apply_crop=True):
    """Capture and process image with optional cropping"""
    # Capture a single frame
    _, frame = cam.read()
    # Convert color space
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if apply_crop:
        # Define your crop coordinates (top left corner and bottom right corner)
        x1, y1 = 90, 0  # Example starting coordinates (top left of the crop rectangle)
        x2, y2 = 600, 480  # Example ending coordinates (bottom right of the crop rectangle)
        # Crop the image
        image = image[y1:y2, x1:x2]
    
    # Resize the image
    image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)
    return image

def capture_images_multi_camera(cameras):
    """Capture images from multiple cameras with camera-specific processing"""
    images = {}
    for cam_name, cam in cameras.items():
        # Apply cropping only for front camera
        apply_crop = (cam_name == 'front')
        images[cam_name] = capture_image(cam, apply_crop=apply_crop)
    return images

def initialize_cameras():
    """Initialize multiple cameras based on config"""
    cameras = {}
    camera_ports = cfg['camera_port']
    camera_names = cfg['camera_names']
    
    # Handle both single camera (backwards compatibility) and multiple cameras
    if isinstance(camera_ports, int):
        # Single camera (old config)
        camera_ports = [camera_ports]
        camera_names = camera_names[:1]  # Use only first camera name
    
    print(f"Initializing {len(camera_ports)} cameras for evaluation...")
    
    for i, (port, name) in enumerate(zip(camera_ports, camera_names)):
        print(f"  Setting up {name} camera on port {port}...")
        cam = cv2.VideoCapture(port)
        
        if not cam.isOpened():
            print(f"❌ Cannot open camera {name} on port {port}")
            # Release any already opened cameras
            for opened_cam in cameras.values():
                opened_cam.release()
            raise IOError(f"Cannot open camera {name} on port {port}")
        
        # Configure camera settings
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        cameras[name] = cam
        print(f"  ✅ {name} camera ready")
    
    return cameras

# Cross-platform audio feedback
def play_sound(type="start"):
    if type == "start":
        # Play a higher pitched beep for start
        winsound.Beep(1000, 500)
        print("\n--- EVALUATION STARTED ---\n")
    else:
        # Play a lower pitched beep for stop
        winsound.Beep(500, 500)
        print("\n--- EVALUATION STOPPED ---\n")


# Helper function to clear the serial buffer
def clear_serial_buffer(ser):
    if ser and ser.in_waiting > 0:
        ser.reset_input_buffer()
        time.sleep(0.1)  # Short delay to ensure buffer is cleared


if __name__ == "__main__":
    
    # Initialize multiple cameras
    cameras = initialize_cameras()
    
    # init follower
    follower = Robot(device_name=ROBOT_PORTS['follower'])
    
    # Clear serial buffer at start
    clear_serial_buffer(follower.ser)
    
    # Set the arm to the initial position
    if not follower.set_initial_position():
        print("⚠️ Failed to set initial position. Continuing anyway...")
    
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
    
    # Start evaluation with sound cue
    play_sound("start")
    
    # Clear buffer right before evaluation
    clear_serial_buffer(follower.ser)


    # bring the follower to the leader
    print("Warming up cameras...")
    for i in range(90):
        follower.read_position()
        _ = capture_images_multi_camera(cameras)
    
    
    obs = {
        'qpos': follower.read_position(),
        'qvel': follower.read_velocity(),
        'images': capture_images_multi_camera(cameras)
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
                
                ### take action
                follower.set_goal_pos(action)

                ### update obs
                obs = {
                    'qpos': follower.read_position(),
                    'qvel': follower.read_velocity(),
                    'images': capture_images_multi_camera(cameras)
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