from config.config_two_cam import TASK_CONFIG, ROBOT_PORTS
import os
import cv2
import h5py
import argparse
from tqdm import tqdm
import winsound 
from time import sleep, time
from training.utils import pwm2pos, pwm2vel

from robot import Robot

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task4')
parser.add_argument('--num_episodes', type=int, default=1)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG

# Cross-platform audio feedback
def play_sound(type="start"):
    if type == "start":
        # Play a higher pitched beep for start
        winsound.Beep(1000, 500)
        print("\n--- RECORDING STARTED ---\n")
    else:
        # Play a lower pitched beep for stop
        winsound.Beep(500, 500)
        print("\n--- RECORDING STOPPED ---\n")

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
    for i, (cam_name, cam) in enumerate(cameras.items()):
        # Apply cropping only for front camera
        apply_crop = (cam_name == 'front')
        images[cam_name] = capture_image(cam, apply_crop=apply_crop)
    return images

# Helper function to clear the serial buffer
def clear_serial_buffer(ser):
    if ser and ser.in_waiting > 0:
        ser.reset_input_buffer()
        sleep(0.1)  # Short delay to ensure buffer is cleared

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
    
    print(f"Initializing {len(camera_ports)} cameras...")
    
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

if __name__ == "__main__":
    # Initialize multiple cameras
    cameras = initialize_cameras()
    
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

    
    for episode in range(num_episodes):
        # Warm up cameras
        print("Warming up cameras...")
        for i in range(50):  # Reduced warm-up time
            _ = capture_images_multi_camera(cameras)
        
        # Start recording with sound cue
        play_sound("start")
        
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
            
            # Capture from all cameras
            images = capture_images_multi_camera(cameras)
            
            obs = {
                'qpos': qpos,  # No conversion needed
                'qvel': qvel,  # No conversion needed
                'images': images  # Now contains all camera images
            }
            # action (leader's position)
            action = leader.read_position()  # Already in radians
            
            # Diagnostic output - show sample joint values
            if show_diagnostics and i % diagnostic_interval == 0:
                print("\n--- Diagnostic Values at Frame", i, "---")
                print(f"Follower Joints: {qpos}")
                print(f"Leader Joints: {action}")
                print(f"Cameras captured: {list(images.keys())}")
                print("-----------------------------------\n")
            
            # store data - no conversion needed
            obs_replay.append(obs)
            action_replay.append(action)

        # End recording with sound cue
        play_sound("stop")

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # Create entries for all cameras
        for cam_name in cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images from all cameras
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
        print(f"Saving episode {episode} with {len(cfg['camera_names'])} cameras...")
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Create datasets for all cameras
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
                
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f"✅ Episode {episode} saved successfully!")
    
    # Cleanup: release all cameras
    print("Releasing cameras...")
    for cam_name, cam in cameras.items():
        cam.release()
        print(f"  Released {cam_name} camera")
    
    leader.set_trigger_torque()
    print("✅ Recording complete!")