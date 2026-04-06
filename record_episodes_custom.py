from config.config import TASK_CONFIG, ROBOT_PORTS
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
parser.add_argument('--task', type=str, default='task7')
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

def capture_image(cam):
    # Capture a single frame
    _, frame = cam.read()
    # Generate a unique filename with the current date and time
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Define your crop coordinates (top left corner and bottom right corner)
    x1, y1 = 90, 0  # Example starting coordinates (top left of the crop rectangle)
    x2, y2 = 600, 480  # Example ending coordinates (bottom right of the crop rectangle)
    # Crop the image
    image = image[y1:y2, x1:x2]
    # Resize the image
    image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)

    return image

# Helper function to clear the serial buffer
def clear_serial_buffer(ser):
    if ser and ser.in_waiting > 0:
        ser.reset_input_buffer()
        sleep(0.1)  # Short delay to ensure buffer is cleared


if __name__ == "__main__":
    # init camera
    cam = cv2.VideoCapture(cfg['camera_port'])
    # Check if the camera opened successfully
    if not cam.isOpened():
        raise IOError("Cannot open camera")
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
            _ = capture_image(cam)
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
            image = capture_image(cam)
            obs = {
                'qpos': qpos,  # No conversion needed
                'qvel': qvel,  # No conversion needed
                'images': {cn: image for cn in cfg['camera_names']}
            }
            # action (leader's position)
            action = leader.read_position()  # Already in radians
            
                # Diagnostic output - show sample joint values
            if show_diagnostics and i % diagnostic_interval == 0:
                print("\n--- Diagnostic Values at Frame", i, "---")
                print(f"Follower Joints: {qpos}")
                print(f"Leader Joints: {action}")
                print("-----------------------------------\n")
            # apply action
            # follower.set_goal_pos(action)  # This is now just a placeholder function
            # store data - no conversion needed
            obs_replay.append(obs)
            action_replay.append(action)

        # End recording with sound cue
        play_sound("stop")

        # disable torque
        leader._disable_torque()
        follower._disable_torque()

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
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
    
    leader.set_trigger_torque()
    # follower._disable_torque()
