from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import warnings
# Suppress verbose PyTorch warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import argparse
import time
import h5py
import numpy as np
import pandas as pd  # For CSV export
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

import alicia_d_sdk  # Use new Alicia SDK instead of old Robot class


# parse the episode file path via command line
parser = argparse.ArgumentParser()
parser.add_argument('--episode_file', type=str, default='/home/hk/Documents/ACT_Shaka/data/task1/episode_12.hdf5',
                   help='Path to the HDF5 episode file to replay')
parser.add_argument('--speed', type=float, default=30.0,
                    help='Robot movement speed in degrees per second')
parser.add_argument('--delay', type=float, default=0.05,
                    help='Delay between commands in seconds')
parser.add_argument('--export-csv', action='store_true', default=True,
                    help='Export XYZ trajectory data to CSV file after replay')
parser.add_argument('--no-export', dest='export_csv', action='store_false',
                    help='Disable CSV export')
args = parser.parse_args()

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG


# Cross-platform audio feedback
def play_sound(type="start"):
    if type == "start":
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("🚀 EPISODE REPLAY STARTED")
        print("="*50 + "\n")
    else:
        # Linux: print visual feedback instead of beep
        print("\n" + "="*50)
        print("✅ EPISODE REPLAY STOPPED")
        print("="*50 + "\n")


def read_robot_state(robot):
    """Read robot joint angles, gripper state, and end-effector XYZ position using Alicia SDK"""
    try:
        # Get joint angles and gripper using unified API
        joint_gripper_state = robot.get_robot_state("joint_gripper")  # Returns JointState object
        
        if joint_gripper_state is None:
            print("Failed to get joint_gripper state")
            return np.zeros(7), np.zeros(7), np.zeros(3)
        
        # Extract joint angles (6 DOF) in radians
        joint_angles = joint_gripper_state.angles  # List of 6 joint angles in radians
        
        # Extract gripper position (0-1000)
        gripper_pos = joint_gripper_state.gripper  # Gripper position 0-1000
        
        # Get end-effector pose using the correct method
        pose_data = robot.get_pose()  # Returns dict with position and quaternion
        
        if pose_data is None:
            print("Failed to get pose data")
            ee_pos = np.zeros(3)
        else:
            # Extract position (xyz in meters)
            ee_pos = np.array(pose_data['position'])  # [x, y, z] in meters
        
        # Combine into 7-DOF state: [joint1, joint2, joint3, joint4, joint5, joint6, gripper_normalized]
        # Convert gripper from 0-1000 to 0-1 range
        gripper_normalized = 1 - gripper_pos / 1000.0
        
        # Create 7-DOF position array
        qpos = joint_angles + [gripper_normalized]
        
        # For velocity, we'll use zeros for now (SDK doesn't provide direct velocity reading in joint_gripper)
        # Could get velocities separately with robot.get_robot_state("velocity") if needed
        qvel = [0.0] * 7
        
        return np.array(qpos), np.array(qvel), ee_pos
    
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return np.zeros(7), np.zeros(7), np.zeros(3)


def set_robot_position(robot, target_pos, speed=30.0):
    """Set robot joint positions using Alicia SDK"""
    try:
        # Split 7-DOF target into 6 joints + gripper
        # Handle both numpy arrays and lists
        if hasattr(target_pos, 'tolist'):
            joint_targets = target_pos[:6].tolist()  # Convert numpy array to list
        else:
            joint_targets = target_pos[:6]  # Already a list
        gripper_target = target_pos[6]   # 7th element is gripper (0-1 normalized)
        
        # Convert gripper from 0-1 to 0-1000 range
        # The policy outputs: 0=closed, 1=open
        # But the robot expects: 0=open, 1000=closed
        # So we need to invert: gripper_raw = (1 - policy_gripper) * 1000
        gripper_target_raw = int((1 - gripper_target) * 1000)
        gripper_target_raw = max(0, min(1000, gripper_target_raw))  # Clamp to valid range
        
        # Set joint angles and gripper using unified method
        success = robot.set_robot_state(
            target_joints=joint_targets,  # 6 joint angles in radians
            gripper_value=gripper_target_raw,  # Gripper 0-1000
            joint_format='rad',  # Input is in radians
            speed_deg_s=speed,  # Configurable speed
            wait_for_completion=False  # Don't wait to maintain replay timing
        )
        
        if not success:
            print(f"Failed to set robot position")
        
    except Exception as e:
        print(f"Error setting robot position: {e}")


def load_episode_data(episode_file):
    """Load qpos data and XYZ positions from HDF5 episode file"""
    try:
        with h5py.File(episode_file, 'r') as f:
            print(f"Loading episode data from: {episode_file}")
            
            # Print dataset info
            print("Available datasets in file:")
            def print_structure(name, obj):
                print(f"  {name}: {obj.shape if hasattr(obj, 'shape') else 'Group'}")
            f.visititems(print_structure)
            
            # Load action data for replay
            if 'action' in f:
                actions = f['action'][:]
                print(f"✓ Loaded {len(actions)} action steps")
                print(f"Action shape: {actions.shape}")
            else:
                print("Error: 'action' dataset not found in file")
                return None, None
            
            # Try to load recorded XYZ positions if available
            recorded_xyz = None
            if 'observations/ee_pos' in f:
                recorded_xyz = f['observations/ee_pos'][:]
                print(f"✓ Loaded {len(recorded_xyz)} recorded XYZ positions")
                print(f"XYZ shape: {recorded_xyz.shape}")
            else:
                print("Note: No recorded XYZ positions found (observations/ee_pos not in file)")
                print("Will compute XYZ from joint angles during replay")
            
            return actions, recorded_xyz
                
    except Exception as e:
        print(f"Error loading episode file: {e}")
        return None, None


def export_xyz_to_csv(recorded_xyz=None, replay_xyz=None, episode_file=""):
    """Export XYZ trajectories to CSV file for analysis"""
    try:
        print("Exporting XYZ trajectory data to CSV...")
        
        # Create timestamp for filename
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Extract episode name from file path for better naming
        episode_name = "unknown"
        if episode_file:
            episode_name = os.path.basename(episode_file).replace('.hdf5', '')
        
        csv_filename = f'xyz_trajectory_{episode_name}_{timestamp}.csv'
        
        # Determine maximum length for synchronization
        max_len = 0
        if recorded_xyz is not None:
            max_len = max(max_len, len(recorded_xyz))
        if replay_xyz is not None:
            replay_xyz = np.array(replay_xyz)
            max_len = max(max_len, len(replay_xyz))
        
        if max_len == 0:
            print("No XYZ data available for export")
            return
        
        # Create data dictionary for CSV
        data = {
            'timestep': list(range(max_len))
        }
        
        # Add recorded XYZ data if available
        if recorded_xyz is not None:
            # Pad with NaN if replay is longer than recording
            recorded_padded = np.full((max_len, 3), np.nan)
            recorded_padded[:len(recorded_xyz)] = recorded_xyz
            
            data['recorded_x'] = recorded_padded[:, 0]
            data['recorded_y'] = recorded_padded[:, 1] 
            data['recorded_z'] = recorded_padded[:, 2]
        else:
            # Add empty columns if no recorded data
            data['recorded_x'] = [np.nan] * max_len
            data['recorded_y'] = [np.nan] * max_len
            data['recorded_z'] = [np.nan] * max_len
        
        # Add replay XYZ data if available
        if replay_xyz is not None and len(replay_xyz) > 0:
            # Pad with NaN if recording is longer than replay
            replay_padded = np.full((max_len, 3), np.nan)
            replay_padded[:len(replay_xyz)] = replay_xyz
            
            data['replay_x'] = replay_padded[:, 0]
            data['replay_y'] = replay_padded[:, 1]
            data['replay_z'] = replay_padded[:, 2]
        else:
            # Add empty columns if no replay data
            data['replay_x'] = [np.nan] * max_len
            data['replay_y'] = [np.nan] * max_len
            data['replay_z'] = [np.nan] * max_len
        
        # Calculate differences if both datasets are available
        # diff = recorded_position[i+1] - actual_replay_position[i]
        # This compares where the recorded trajectory is going next vs where replay currently is
        # Positive diff means replay needs to move in positive direction to follow trajectory
        # Negative diff means replay needs to move in negative direction to follow trajectory
        if recorded_xyz is not None and replay_xyz is not None and len(replay_xyz) > 0:
            # We need at least 2 recorded points and 1 replay point for this calculation
            min_len = min(len(recorded_xyz) - 1, len(replay_xyz))  # -1 because we use i+1 indexing
            
            if min_len > 0:
                # Calculate error: recorded[i+1] - actual_replay[i] for each axis
                diff_x = recorded_xyz[1:min_len+1, 0] - replay_xyz[:min_len, 0]  # recorded_x[i+1] - actual_x[i]
                diff_y = recorded_xyz[1:min_len+1, 1] - replay_xyz[:min_len, 1]  # recorded_y[i+1] - actual_y[i]  
                diff_z = recorded_xyz[1:min_len+1, 2] - replay_xyz[:min_len, 2]  # recorded_z[i+1] - actual_z[i]
            else:
                diff_x = np.array([])
                diff_y = np.array([])
                diff_z = np.array([])
            
            # Pad differences with NaN for remaining timesteps
            # First row is ignored (NaN), then differences start from timestep 1
            data['diff_x'] = [np.nan] * max_len
            data['diff_y'] = [np.nan] * max_len
            data['diff_z'] = [np.nan] * max_len
            
            if len(diff_x) > 0:
                # Start filling from index 1 (skip first row as requested)
                data['diff_x'][1:min_len+1] = diff_x.tolist()
                data['diff_y'][1:min_len+1] = diff_y.tolist()  
                data['diff_z'][1:min_len+1] = diff_z.tolist()
                
                # Calculate distance error
                distance_error = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                data['distance_error'] = [np.nan] * max_len
                data['distance_error'][1:min_len+1] = distance_error.tolist()
            else:
                data['distance_error'] = [np.nan] * max_len
        else:
            data['diff_x'] = [np.nan] * max_len
            data['diff_y'] = [np.nan] * max_len
            data['diff_z'] = [np.nan] * max_len
            data['distance_error'] = [np.nan] * max_len
        
        # Create DataFrame and save to CSV
        try:
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False, float_format='%.6f')
        except Exception as e:
            print(f"Pandas not available, using basic CSV export: {e}")
            # Fallback: basic CSV export without pandas
            import csv
            
            with open(csv_filename, 'w', newline='') as csvfile:
                # Write header
                headers = ['timestep', 'recorded_x', 'recorded_y', 'recorded_z', 
                          'replay_x', 'replay_y', 'replay_z', 
                          'diff_x', 'diff_y', 'diff_z', 'distance_error']
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                # Write data rows
                for i in range(max_len):
                    row = [
                        i,
                        f"{data['recorded_x'][i]:.6f}" if not np.isnan(data['recorded_x'][i]) else "",
                        f"{data['recorded_y'][i]:.6f}" if not np.isnan(data['recorded_y'][i]) else "",
                        f"{data['recorded_z'][i]:.6f}" if not np.isnan(data['recorded_z'][i]) else "",
                        f"{data['replay_x'][i]:.6f}" if not np.isnan(data['replay_x'][i]) else "",
                        f"{data['replay_y'][i]:.6f}" if not np.isnan(data['replay_y'][i]) else "",
                        f"{data['replay_z'][i]:.6f}" if not np.isnan(data['replay_z'][i]) else "",
                        f"{data['diff_x'][i]:.6f}" if not np.isnan(data['diff_x'][i]) else "",
                        f"{data['diff_y'][i]:.6f}" if not np.isnan(data['diff_y'][i]) else "",
                        f"{data['diff_z'][i]:.6f}" if not np.isnan(data['diff_z'][i]) else "",
                        f"{data['distance_error'][i]:.6f}" if not np.isnan(data['distance_error'][i]) else ""
                    ]
                    writer.writerow(row)
        
        print(f"✓ XYZ trajectory data exported to: {csv_filename}")
        
        # Print summary statistics
        if recorded_xyz is not None and replay_xyz is not None and len(replay_xyz) > 0:
            min_len = min(len(recorded_xyz) - 1, len(replay_xyz))  # -1 because we use i+1 indexing
            if min_len > 0:
                print(f"\nTrajectory Prediction Analysis:")
                print(f"Compared timesteps: {min_len} (starting from timestep 1, first row ignored)")
                print(f"Error = Recorded[i+1] - Actual_Replay[i] (trajectory following accuracy)")
                print(f"RMS Error - X: {np.sqrt(np.mean(diff_x**2)):.6f}m (lateral following)")
                print(f"RMS Error - Y: {np.sqrt(np.mean(diff_y**2)):.6f}m (forward/back following)") 
                print(f"RMS Error - Z: {np.sqrt(np.mean(diff_z**2)):.6f}m (vertical following)")
                print(f"Mean Distance Error: {np.mean(distance_error):.6f}m (3D following accuracy)")
                print(f"Max Distance Error: {np.max(distance_error):.6f}m (worst following error)")
                
                # Additional statistics
                print(f"Mean Bias - X: {np.mean(diff_x):.6f}m ({'needs +X movement' if np.mean(diff_x) > 0 else 'needs -X movement'})")
                print(f"Mean Bias - Y: {np.mean(diff_y):.6f}m ({'needs +Y movement' if np.mean(diff_y) > 0 else 'needs -Y movement'})")
                print(f"Mean Bias - Z: {np.mean(diff_z):.6f}m ({'needs +Z movement' if np.mean(diff_z) > 0 else 'needs -Z movement'})")
            else:
                print(f"\nInsufficient data for trajectory prediction analysis")
        else:
            print(f"\nNo data available for trajectory analysis")
        
        return csv_filename
        
    except Exception as e:
        print(f"Error exporting XYZ data to CSV: {e}")
        return None


if __name__ == "__main__":
    # Check if episode file exists
    if not os.path.exists(args.episode_file):
        print(f"Error: Episode file not found: {args.episode_file}")
        exit(1)
    
    # Load episode data and recorded XYZ positions
    qpos_data, recorded_xyz = load_episode_data(args.episode_file)
    if qpos_data is None:
        print("Failed to load episode data")
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
    
    # Get initial robot state
    print("Reading initial robot state...")
    qpos, qvel, initial_xyz = read_robot_state(follower)
    print(f"Initial robot position: {qpos}")
    print(f"Initial XYZ position: {initial_xyz}")
    
    # Prepare for replay
    print(f"\nPreparing to replay {len(qpos_data)} qpos steps...")
    print(f"Speed: {args.speed} deg/s, Delay: {args.delay}s between commands")
    print("Starting replay in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    # Start replay with sound cue
    play_sound("start")
    
    # Initialize lists to store replay data for comparison
    actual_positions = []
    replay_xyz_positions = []
    
    try:
        # Replay episode qpos data
        for i, qpos in enumerate(tqdm(qpos_data, desc="Replaying episode")):
            # Convert qpos to proper format
            if isinstance(qpos, np.ndarray):
                qpos = qpos.astype(float).tolist()
            
            # Set robot position using qpos data
            set_robot_position(follower, qpos, speed=args.speed)
            
            # Read XYZ position immediately after sending command (EVERY step, not sampling)
            try:
                current_qpos, _, current_xyz = read_robot_state(follower)
                actual_positions.append(current_qpos.copy())
                replay_xyz_positions.append(current_xyz.copy())
                
                # Print debug info less frequently to avoid spam
                if i % 50 == 0:  # Print every 50th step only for debugging
                    print(f"Step {i}: Target={np.array(qpos)[:3]}, Current={current_qpos[:3]}")
                    print(f"         XYZ: {current_xyz}")
            except Exception as e:
                # Don't let data collection errors affect the main replay timing
                print(f"Warning: Failed to read robot state at step {i}: {e}")
                # Add placeholder data to keep arrays consistent
                if len(actual_positions) > 0:
                    actual_positions.append(actual_positions[-1].copy())
                    replay_xyz_positions.append(replay_xyz_positions[-1].copy())
                else:
                    actual_positions.append(np.zeros(7))
                    replay_xyz_positions.append(np.zeros(3))
            
            # Maintain original timing between commands
            time.sleep(args.delay)
            
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    
    except Exception as e:
        print(f"Error during replay: {e}")
    
    finally:
        # End replay with sound cue
        play_sound("stop")
        
        # Print data collection summary
        print(f"\nData Collection Summary:")
        print(f"Episode commands: {len(qpos_data)}")
        print(f"Recorded XYZ points: {len(recorded_xyz) if recorded_xyz is not None else 0}")
        print(f"Replay XYZ points: {len(replay_xyz_positions)}")
        print(f"Data collection rate: {len(replay_xyz_positions) / len(qpos_data) * 100:.1f}%")
        
        # Export XYZ trajectories to CSV if requested
        if args.export_csv:
            print("\nExporting XYZ trajectory data to CSV...")
            csv_file = export_xyz_to_csv(recorded_xyz, replay_xyz_positions, args.episode_file)
            if csv_file:
                print(f"✓ Data exported successfully to {csv_file}")
            else:
                print("✗ CSV export failed")
        else:
            print("\nSkipping CSV export (disabled)")
        
        # Disconnect robot properly
        print("Disconnecting robot...")
        try:
            follower.disconnect()
            print("✓ Robot disconnected")
        except Exception as e:
            print(f"Error disconnecting robot: {e}")
        
        print("✅ Episode replay completed")