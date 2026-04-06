"""
Real Robot End-Effector Calibration Tool
Records qpos-to-EE xyz mapping for the real Alicia robot arm.
Outputs JSON file compatible with simulation calibration data format.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from time import sleep
import cv2

from config.config import TASK_CONFIG, ROBOT_PORTS
import alicia_d_sdk  # Use Alicia SDK
import pyrealsense2 as rs  # RealSense SDK for L515 camera

def get_real_ee_pose(robot):
    """
    Get current end-effector pose (xyz position) from real robot.
    Returns xyz position as numpy array [x, y, z].
    """
    try:
        # Get end-effector pose using Alicia SDK get_pose() method
        pose_dict = robot.get_pose()  # Returns dict with position, rotation, etc.
        
        if pose_dict is None:
            print("Failed to get robot pose")
            return np.zeros(3)
        
        # Extract position (x, y, z) from pose dictionary
        # pose_dict['position'] should be array-like with [x, y, z] in meters
        position = pose_dict['position']
        ee_xyz = np.array(position[:3])  # Take first 3 elements (x, y, z)
        
        return ee_xyz
        
    except Exception as e:
        print(f"Error getting real EE pose: {e}")
        return np.zeros(3)

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
        qvel = [0.0] * 7
        
        return np.array(qpos), np.array(qvel)
    
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return np.zeros(7), np.zeros(7)

def capture_image(pipeline):
    """Capture and resize camera frame using RealSense SDK"""
    try:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("Warning: Failed to capture camera frame")
            # Return a black frame if capture fails
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert to numpy array (RealSense returns BGR format)
        frame = np.asanyarray(color_frame.get_data())
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to standard resolution
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

        return image
    
    except Exception as e:
        print(f"Error capturing image: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

def real_robot_calibration_mode(robot, camera_pipeline, output_dir):
    """
    Interactive calibration mode to record qpos-to-EE mapping for REAL robot.
    Press SPACE to record current pose, Q to quit and save data.
    """
    print("\n" + "="*60)
    print("REAL ROBOT KINEMATIC CALIBRATION MODE")
    print("="*60)
    print("Instructions:")
    print("- Move robot to different representative positions manually")  
    print("- Press SPACE to record current qpos + EE position")
    print("- Press Q to quit and save calibration data")
    print("- The goal is to collect qpos->EE mapping for real robot")
    print("="*60 + "\n")
    
    calibration_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'qpos to end-effector xyz mapping for REAL robot calibration',
        'robot_model': 'alicia_real',
        'robot_type': 'physical_hardware',
        'poses': []
    }
    
    pose_count = 0
    
    # Setup rendering for visual feedback
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize camera image display
    initial_image = capture_image(camera_pipeline)
    plt_img = ax1.imshow(initial_image)
    ax1.set_title("Real Robot Camera View")
    ax1.axis('off')
    
    # Initialize trajectory plot
    ax2.set_title("Recorded EE Trajectory (XY)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.ion()
    plt.tight_layout()
    
    print("Ready for calibration! Move robot to different poses...")
    print("Use keyboard: SPACE=record, Q=quit")
    
    # Lists to store trajectory data for plotting
    x_positions = []
    y_positions = []
    pose_ids = []
    
    # Simplified keyboard handling with matplotlib
    def on_key(event):
        nonlocal pose_count, calibration_data, x_positions, y_positions, pose_ids
        
        if event.key == ' ':  # Space bar
            # Record current pose
            current_qpos, _ = read_robot_state(robot)
            current_ee_xyz = get_real_ee_pose(robot)
            
            pose_data = {
                'pose_id': pose_count,
                'qpos': current_qpos.tolist(),  # 7-DOF joint positions
                'ee_xyz': current_ee_xyz.tolist(),  # [x, y, z] end-effector position
                'timestamp': datetime.now().isoformat()
            }
            
            calibration_data['poses'].append(pose_data)
            
            # Store for trajectory plotting
            x_positions.append(current_ee_xyz[0])
            y_positions.append(current_ee_xyz[1])
            pose_ids.append(pose_count)
            
            pose_count += 1
            
            print(f"✓ Recorded pose {pose_count}:")
            print(f"  qpos: {current_qpos}")
            print(f"  EE xyz: {current_ee_xyz}")
            print(f"  Total poses: {pose_count}")
            
            # Update trajectory plot
            ax2.clear()
            ax2.set_title(f"Recorded EE Trajectory (XY) - {pose_count} poses")
            ax2.set_xlabel("X Position (m)")
            ax2.set_ylabel("Y Position (m)")
            ax2.grid(True)
            if len(x_positions) > 1:
                ax2.plot(x_positions, y_positions, 'b-o', markersize=8, linewidth=2)
                # Add pose ID labels
                for i, (x, y, pid) in enumerate(zip(x_positions, y_positions, pose_ids)):
                    ax2.annotate(f'{pid}', (x, y), xytext=(5, 5), textcoords='offset points')
            elif len(x_positions) == 1:
                ax2.plot(x_positions[0], y_positions[0], 'ro', markersize=10)
                ax2.annotate(f'{pose_ids[0]}', (x_positions[0], y_positions[0]), xytext=(5, 5), textcoords='offset points')
            ax2.set_aspect('equal')
            
        elif event.key == 'q':
            # Save and quit
            save_real_calibration_data(calibration_data, pose_count, output_dir)
            plt.close()
            print("Real robot calibration completed!")
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    try:
        print("Starting real robot calibration loop...")
        while True:
            # Update camera image display
            current_image = capture_image(camera_pipeline)
            plt_img.set_data(current_image)
            
            # Update title with pose count
            ax1.set_title(f"Real Robot Camera - Poses recorded: {pose_count} (SPACE=record, Q=quit)")
            
            plt.pause(0.1)  # Slower update rate for real robot
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Save data if any poses recorded
        if pose_count > 0:
            save_real_calibration_data(calibration_data, pose_count, output_dir)
        plt.close()

def save_real_calibration_data(calibration_data, pose_count, output_dir):
    """Save real robot calibration data to JSON file"""
    if pose_count > 0:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        calib_filename = f"real_robot_calibration_{timestamp_str}.json"
        calib_path = os.path.join(output_dir, calib_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        with open(calib_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✓ Real robot calibration data saved to: {calib_path}")
        print(f"✓ Recorded {pose_count} pose mappings")
        print(f"✓ Data format: qpos (7-DOF) -> ee_xyz (3D position)")
        print(f"✓ Robot type: Real hardware (Alicia)")
        
        # Print summary statistics
        if pose_count > 1:
            poses = calibration_data['poses']
            x_coords = [pose['ee_xyz'][0] for pose in poses]
            y_coords = [pose['ee_xyz'][1] for pose in poses]
            z_coords = [pose['ee_xyz'][2] for pose in poses]
            
            print(f"\n--- EE Position Statistics ---")
            print(f"X range: {min(x_coords):.4f} to {max(x_coords):.4f} m")
            print(f"Y range: {min(y_coords):.4f} to {max(y_coords):.4f} m") 
            print(f"Z range: {min(z_coords):.4f} to {max(z_coords):.4f} m")
    else:
        print("\nNo poses recorded. Real robot calibration data not saved.")

def main():
    parser = argparse.ArgumentParser(description='Record real robot end-effector calibration data')
    parser.add_argument('--output_dir', type=str, default='./real_robot_calibration', 
                       help='Directory to save calibration data')
    parser.add_argument('--robot_port', type=str, default=None,
                       help='Robot port (if None, uses ROBOT_PORTS leader from config)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize L515 camera
    print("Initializing L515 camera with RealSense SDK...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream - use fixed 640x480 resolution
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        profile = pipeline.start(config)
        print("✓ L515 camera started successfully")
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        
        # Allow camera to settle
        for i in range(30):
            pipeline.wait_for_frames()
            
    except Exception as e:
        print(f"Failed to start L515 camera: {e}")
        print("Make sure the camera is connected and not being used by another application")
        return
    
    # Initialize real robot
    print("Initializing real robot...")
    try:
        robot_port = args.robot_port if args.robot_port else ROBOT_PORTS['leader']
        robot = alicia_d_sdk.create_robot(
            port=robot_port,
            gripper_type="50mm"
        )
        print(f"✓ Real robot connected successfully on port {robot_port}")
        
        # Test robot state reading
        test_qpos, _ = read_robot_state(robot)
        test_ee_xyz = get_real_ee_pose(robot)
        print(f"✓ Robot state reading test successful")
        print(f"  Current qpos: {test_qpos}")
        print(f"  Current EE xyz: {test_ee_xyz}")
        
    except Exception as e:
        print(f"Failed to connect real robot: {e}")
        print("Make sure the robot is connected and powered on")
        pipeline.stop()
        return
    
    try:
        # Start calibration mode
        print(f"\nStarting real robot calibration mode...")
        print(f"Move the robot manually to different representative poses")
        print(f"Press SPACE in the plot window to record poses")
        print(f"Press Q to quit and save data")
        
        real_robot_calibration_mode(robot, pipeline, output_dir)
        
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            robot.disconnect()
            print("✓ Robot disconnected")
        except:
            pass
        
        try:
            pipeline.stop()
            print("✓ Camera stopped")
        except:
            pass
        
        print("✓ Real robot calibration completed")

if __name__ == "__main__":
    main()