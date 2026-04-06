from config.config_depth_cam import TASK_CONFIG, ROBOT_PORTS
import os
import cv2
import h5py
import argparse
from tqdm import tqdm
from time import sleep, time 
from training.utils import pwm2pos, pwm2vel
import pyrealsense2 as rs
import numpy as np

from robot import Robot

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task8')
parser.add_argument('--num_episodes', type=int, default=1)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG

def capture_image(pipeline, align_to_color, depth_scale, depth_offset_mm):
    """
    Capture RGBD image from L515 camera.
    Returns (H, W, 4) array with [R, G, B, D_normalized] for ACT training.
    Uses exact same logic as test_l515_advanced.py
    """
    # Wait for frames
    frames = pipeline.wait_for_frames()
    
    # Align depth to color
    aligned_frames = align_to_color.process(frames)
    
    color_frame = aligned_frames.get_color_frame()
    aligned_depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not aligned_depth_frame:
        raise RuntimeError("Failed to capture RGBD frames")
    
    # Convert to numpy
    color_image_bgr = np.asanyarray(color_frame.get_data())  # BGR format from camera
    depth_data_raw = np.asanyarray(aligned_depth_frame.get_data())
    
    # Convert depth to mm (same as test_l515_advanced.py)
    h, w = depth_data_raw.shape
    center_dist_api = aligned_depth_frame.get_distance(w//2, h//2) * 1000  # mm
    aligned_depth_image = depth_data_raw.astype(np.float32) * depth_scale * 1000  # to mm
    center_dist_calc = aligned_depth_image[h//2, w//2]
    
    # Check if offset compensation needed (same check as test_l515_advanced.py)
    if abs(center_dist_api - center_dist_calc) > 10:  # More than 10mm difference
        aligned_depth_image = aligned_depth_image - depth_offset_mm
        aligned_depth_image[aligned_depth_image < 0] = 0
    
    # Convert BGR to RGB for storage/training
    color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    
    # Crop (same as before)
    x1, y1 = 90, 0
    x2, y2 = 600, 480
    color_cropped = color_image_rgb[y1:y2, x1:x2]
    depth_cropped = aligned_depth_image[y1:y2, x1:x2]
    
    # Resize
    color_resized = cv2.resize(color_cropped, (cfg['cam_width'], cfg['cam_height']), 
                               interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_cropped, (cfg['cam_width'], cfg['cam_height']), 
                               interpolation=cv2.INTER_NEAREST)  # Use NEAREST for depth
    
    # Normalize depth to 0-255 range for CNN input
    # Typical workspace: 200mm (20cm) to 1500mm (1.5m)
    depth_min_mm = 200.0
    depth_max_mm = 1500.0
    depth_normalized = np.clip(
        (depth_resized - depth_min_mm) / (depth_max_mm - depth_min_mm) * 255.0,
        0, 255
    ).astype(np.uint8)
    
    # Create RGBD array: [R, G, B, D_normalized]
    rgbd = np.concatenate([color_resized, depth_normalized[:, :, np.newaxis]], axis=2)
    
    return rgbd

# Helper function to clear the serial buffer
def clear_serial_buffer(ser):
    if ser and ser.in_waiting > 0:
        ser.reset_input_buffer()
        sleep(0.1)  # Short delay to ensure buffer is cleared


if __name__ == "__main__":
    # Initialize L515 camera (same setup as test_l515_advanced.py)
    print("Initializing Intel RealSense L515...")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams (exact same as test_l515_advanced.py)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)  # Native resolution
    
    # Start pipeline
    profile = pipeline.start(config)
    
    # Get depth sensor and configure
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # Set Short Range preset for close-range work
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # Short Range
        print("✓ Set to Short Range preset (min ~25cm)")
    
    # Get depth scale
    depth_scale = depth_sensor.get_depth_scale()
    print(f"✓ Depth scale: {depth_scale}")
    
    # Check depth offset
    depth_offset_mm = 0
    if depth_sensor.supports(rs.option.depth_offset):
        depth_offset_meters = depth_sensor.get_option(rs.option.depth_offset)
        depth_offset_mm = depth_offset_meters * 1000
        if abs(depth_offset_mm) > 1:
            print(f"⚠️  Depth offset detected: {depth_offset_mm:.1f}mm (will be compensated)")
    
    # Create align object
    align_to_color = rs.align(rs.stream.color)
    
    print(f"✓ L515 initialized for RGBD capture")
    print(f"  Resolution: 640x480")
    print(f"  Output: {cfg['cam_width']}x{cfg['cam_height']}x4 (RGBD)")
    
    # Warm up camera
    print("Warming up camera...")
    for _ in range(30):
        frames = pipeline.wait_for_frames()
    print("✓ Ready!")
    
    # init follower
    follower = Robot(device_name=ROBOT_PORTS['follower'])
    # init leader
    leader = Robot(device_name=ROBOT_PORTS['leader'])
    leader._disable_torque()
    
        # Clear both serial buffers at start
    clear_serial_buffer(leader.ser)
    clear_serial_buffer(follower.ser)
    
    # Wait for stable communication
    print("Waiting for stable communication...")
    sleep(2)

    
    for i in range(num_episodes):
        # bring the follower to the leader and start camera
        for i in range(200):
            _ = capture_image(pipeline, align_to_color, depth_scale, depth_offset_mm)
        
        # Start recording
        print("\n" + "="*60)
        print("🔴 RECORDING STARTED")
        print("="*60 + "\n")
        
        # Clear buffers again right before recording
        clear_serial_buffer(leader.ser)
        clear_serial_buffer(follower.ser)
        
        # init buffers
        obs_replay = []
        action_replay = []
        
        # Add diagnostic counters
        diagnostic_interval = 30  # Show values every 30 frames
        show_diagnostics = True

        for i in tqdm(range(cfg['episode_len'])):
            # observation
            qpos = follower.read_position()  # Already in radians
            qvel = follower.read_velocity()  # Already in radians (or zeros)
            ee_pos = follower.read_ee_position()  # End-effector position [x, y, z] in mm
            image = capture_image(pipeline, align_to_color, depth_scale, depth_offset_mm)
            obs = {
                'qpos': qpos,  # No conversion needed
                'qvel': qvel,  # No conversion needed
                'ee_pos': ee_pos if ee_pos is not None else np.zeros(3),  # Robot frame position
                'images': {cn: image for cn in cfg['camera_names']}
            }
            # action (leader's position)
            action = leader.read_position()  # Already in radians
            
                # Diagnostic output - show sample joint values
            if show_diagnostics and i % diagnostic_interval == 0:
                print("\n--- Diagnostic Values at Frame", i, "---")
                print(f"Follower Joints: {qpos}")
                print(f"Follower EE Pos (mm): {obs['ee_pos']}")
                print(f"Leader Joints: {action}")
                print("-----------------------------------\n")
            # apply action
            # follower.set_goal_pos(action)  # This is now just a placeholder function
            # store data - no conversion needed
            obs_replay.append(obs)
            action_replay.append(action)

        # End recording
        print("\n" + "="*60)
        print("⏹️  RECORDING STOPPED")
        print("="*60 + "\n")

        # disable torque
        #leader._disable_torque()
        #follower._disable_torque()

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/ee_pos': [],  # End-effector position for R2RealGen
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/observations/ee_pos'].append(o['ee_pos'])  # Store EE position
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
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                # RGBD: 4 channels instead of 3
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 4), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 4), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            ee_pos = obs.create_dataset('ee_pos', (max_timesteps, 3))  # [x, y, z] in mm
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
    
    leader.set_trigger_torque()
    # follower._disable_torque()
    
    # Stop pipeline
    pipeline.stop()
    print("\n✓ L515 pipeline stopped")
