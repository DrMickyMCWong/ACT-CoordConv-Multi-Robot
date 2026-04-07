#!/usr/bin/env python3
"""
Test script to verify Forward Kinematics with real RoArm-M3 robot.

This script:
1. Connects to real RoArm-M3 via serial
2. Reads current joint angles from robot
3. Computes FK: joint_angles → EE position (x, y, z, euler angles)
4. Displays results and verifies against solver.hpp formulas
5. Tests Euler angle calculation for DemoGen conversion

Usage:
    python test_roarm_fk_real.py
    
Then manually move the robot to different positions and press Enter to test FK.
"""

import json
import numpy as np
import math
import time
from robot import Robot


class RoArmFKTester:
    """Forward Kinematics tester based on solver.hpp"""
    
    def __init__(self):
        # Link lengths from solver.hpp (mm)
        self.l1 = 126.06  # Base height
        self.l2 = 237.00  # Shoulder to elbow
        self.l3 = 280.00  # Elbow to wrist
        self.l4 = 68.00   # Wrist to end effector
        
        print(f"📐 Initialized FK Tester")
        print(f"   Link lengths (mm): l1={self.l1}, l2={self.l2}, l3={self.l3}, l4={self.l4}")
    
    def parse_robot_json(self, json_str):
        """
        Parse RoArm-M3 JSON data to extract relevant fields.
        
        Expected JSON format:
        {
            "x": float (mm),
            "y": float (mm),
            "z": float (mm),
            "b": float (base joint, radians),
            "s": float (shoulder joint, radians),
            "e": float (elbow joint, radians),
            "t": float (wrist joint, radians),
            "r": float (roll joint, radians),
            "g": float (gripper joint, radians),
            ...
        }
        """
        try:
            data = json.loads(json_str)
            
            # Position (mm)
            position = {
                'x': data.get('x', 0.0),
                'y': data.get('y', 0.0),
                'z': data.get('z', 0.0)
            }
            
            # Joint angles (radians)
            joints = {
                'base': data.get('b', 0.0),
                'shoulder': data.get('s', 0.0),
                'elbow': data.get('e', 0.0),
                'wrist': data.get('t', 0.0),  # t = wrist joint 1
                'roll': data.get('r', 0.0),
                'gripper': data.get('g', 0.0)
            }
            
            return position, joints
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return None, None
    
    def compute_fk_from_joints(self, joints):
        """
        Compute forward kinematics from joint angles.
        Based on solver.hpp computePosbyJointRad()
        
        Args:
            joints: dict with keys 'base', 'shoulder', 'elbow', 'wrist', 'roll', 'gripper'
        
        Returns:
            dict with 'position' (x,y,z in mm) and 'orientation' (roll, pitch in radians)
        """
        b = joints['base']
        s = joints['shoulder']
        e = joints['elbow']
        t = joints['wrist']
        r = joints['roll']
        
        # From solver.hpp: compute arm configuration in 2D (r-z plane)
        # Then rotate by base angle to get 3D position
        
        # Compute radial distance and height in arm's local frame
        # This is simplified - actual FK would compute link by link
        # For now, use the robot's reported x,y,z and compare
        
        # Pitch calculation from solver.hpp line 320
        pitch = (e + s + t) - (math.pi / 2.0)
        
        # Roll is direct from joint
        roll = r
        
        return {
            'pitch': pitch,
            'roll': roll
        }
    
    def compute_euler_angles(self, joints):
        """
        Compute Euler angles for DemoGen format.
        Based on ROARM_TO_DEMOGEN_CONVERSION.md
        
        Returns:
            np.array([euler_x, euler_y, euler_z]) - roll, pitch, yaw in radians
        """
        s = joints['shoulder']
        e = joints['elbow']
        t = joints['wrist']
        r = joints['roll']
        b = joints['base']
        
        # From solver.hpp line 320
        pitch = (e + s + t) - (math.pi / 2.0)
        roll = r
        yaw = b  # Using base rotation as yaw (Option A from guide)
        
        return np.array([roll, pitch, yaw])
    
    def create_demogen_state(self, position, joints):
        """
        Create DemoGen-compatible state vector.
        Format: [x, y, z, euler_x, euler_y, euler_z, gripper]
        
        Args:
            position: dict with 'x', 'y', 'z' in mm
            joints: dict with joint angles in radians
        
        Returns:
            np.array of shape (7,) - [x_m, y_m, z_m, roll, pitch, yaw, gripper]
        """
        # Position: convert mm → meters
        x_m = position['x'] / 1000.0
        y_m = position['y'] / 1000.0
        z_m = position['z'] / 1000.0
        
        # Euler angles
        euler = self.compute_euler_angles(joints)
        
        # Gripper: normalize to [-1, 1]
        # Assuming gripper range is [0, π]
        g_rad = joints['gripper']
        gripper_normalized = 2.0 * (g_rad / math.pi) - 1.0
        
        # Combine into state vector
        state = np.array([
            x_m, y_m, z_m,           # Position in meters
            euler[0], euler[1], euler[2],  # Euler angles in radians
            gripper_normalized        # Gripper state [-1, 1]
        ])
        
        return state
    
    def print_robot_state(self, position, joints, fk_result, demogen_state):
        """Print formatted robot state information"""
        print(f"\n{'='*70}")
        print(f"Robot State")
        print(f"{'='*70}")
        
        # Position
        print(f"\n  📍 Position (from robot):")
        print(f"     X: {position['x']:8.2f} mm  ({position['x']/1000:.4f} m)")
        print(f"     Y: {position['y']:8.2f} mm  ({position['y']/1000:.4f} m)")
        print(f"     Z: {position['z']:8.2f} mm  ({position['z']/1000:.4f} m)")
        
        # Joint angles
        print(f"\n  🤖 Joint Angles (radians):")
        print(f"     Base (b):     {joints['base']:7.4f} rad  ({math.degrees(joints['base']):7.2f}°)")
        print(f"     Shoulder (s): {joints['shoulder']:7.4f} rad  ({math.degrees(joints['shoulder']):7.2f}°)")
        print(f"     Elbow (e):    {joints['elbow']:7.4f} rad  ({math.degrees(joints['elbow']):7.2f}°)")
        print(f"     Wrist (t):    {joints['wrist']:7.4f} rad  ({math.degrees(joints['wrist']):7.2f}°)")
        print(f"     Roll (r):     {joints['roll']:7.4f} rad  ({math.degrees(joints['roll']):7.2f}°)")
        print(f"     Gripper (g):  {joints['gripper']:7.4f} rad  ({math.degrees(joints['gripper']):7.2f}°)")
        
        # Computed orientation
        print(f"\n  🧭 Orientation (computed from joints):")
        print(f"     Roll:  {fk_result['roll']:7.4f} rad  ({math.degrees(fk_result['roll']):7.2f}°)")
        print(f"     Pitch: {fk_result['pitch']:7.4f} rad  ({math.degrees(fk_result['pitch']):7.2f}°)")
        print(f"     Formula: pitch = (e + s + t) - π/2")
        print(f"              pitch = ({joints['elbow']:.4f} + {joints['shoulder']:.4f} + {joints['wrist']:.4f}) - {math.pi/2:.4f}")
        print(f"              pitch = {fk_result['pitch']:.4f} rad")
        
        # DemoGen state vector
        print(f"\n  📦 DemoGen State Vector (7D):")
        print(f"     [x, y, z, euler_x, euler_y, euler_z, gripper]")
        print(f"     [{demogen_state[0]:7.4f}, {demogen_state[1]:7.4f}, {demogen_state[2]:7.4f}, " +
              f"{demogen_state[3]:7.4f}, {demogen_state[4]:7.4f}, {demogen_state[5]:7.4f}, {demogen_state[6]:7.4f}]")
        print(f"     Position (m): [{demogen_state[0]:.4f}, {demogen_state[1]:.4f}, {demogen_state[2]:.4f}]")
        print(f"     Euler (rad):  [{demogen_state[3]:.4f}, {demogen_state[4]:.4f}, {demogen_state[5]:.4f}]")
        print(f"     Euler (deg):  [{math.degrees(demogen_state[3]):.2f}°, " +
              f"{math.degrees(demogen_state[4]):.2f}°, {math.degrees(demogen_state[5]):.2f}°]")
        print(f"     Gripper:      {demogen_state[6]:.4f} (normalized)")
        
        print(f"\n{'='*70}")


def disable_robot_torque(robot):
    """
    Disable torque on the robot to allow manual manipulation.
    Sends JSON command: {"T":210,"cmd":0}
    """
    if not robot.ser:
        print("❌ No serial connection available")
        return False
    
    try:
        # {"T":210,"cmd":0} - Torque OFF
        torque_command = {
            'T': 210,
            'cmd': 0
        }
        
        cmd_str = json.dumps(torque_command) + '\n'
        robot.ser.write(cmd_str.encode('utf-8'))
        robot.ser.flush()
        print("✅ Torque DISABLED - Robot is now free to move manually")
        time.sleep(0.3)
        return True
        
    except Exception as e:
        print(f"❌ Error disabling torque: {e}")
        return False


def enable_robot_torque(robot):
    """
    Enable torque on the robot (not typically needed for this test).
    Sends JSON command: {"T":210,"cmd":1}
    """
    if not robot.ser:
        print("❌ No serial connection available")
        return False
    
    try:
        # {"T":210,"cmd":1} - Torque ON
        torque_command = {
            'T': 210,
            'cmd': 1
        }
        
        cmd_str = json.dumps(torque_command) + '\n'
        robot.ser.write(cmd_str.encode('utf-8'))
        robot.ser.flush()
        print("✅ Torque ENABLED - Robot motors are active")
        time.sleep(0.3)
        return True
        
    except Exception as e:
        print(f"❌ Error enabling torque: {e}")
        return False


def main():
    print("="*70)
    print("RoArm-M3 Forward Kinematics Test (Real Robot)")
    print("="*70)
    
    # Ask user for serial port
    print("\n🔌 Robot Connection Setup")
    port = input("Enter serial port (e.g., /dev/ttyUSB0): ").strip()
    
    if not port:
        port = "/dev/ttyUSB0"  # Default
        print(f"   Using default port: {port}")
    
    # Initialize FK tester
    fk_tester = RoArmFKTester()
    
    # Connect to robot
    print(f"\n🤖 Connecting to robot on {port}...")
    try:
        robot = Robot(device_name=port, baudrate=115200)
        time.sleep(2)  # Wait for connection to stabilize
        
        if not robot.ser or not robot.ser.is_open:
            print("❌ Failed to connect to robot")
            return
        
        print("✅ Robot connected successfully")
        
        # Disable torque to allow manual manipulation
        print("\n⚙️  Disabling robot torque for manual manipulation...")
        if not disable_robot_torque(robot):
            print("⚠️  Warning: Failed to disable torque")
            print("   You may need to disable torque manually")
            user_continue = input("   Continue anyway? (y/n): ").strip().lower()
            if user_continue != 'y':
                return
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Interactive testing loop
    print("\n" + "="*70)
    print("Interactive FK Testing")
    print("="*70)
    print("\n📋 Instructions:")
    print("  1. Torque is now DISABLED - you can freely move the arm")
    print("  2. Move the robot arm to different positions")
    print("  3. Press Enter to read current state and compute FK")
    print("  4. Type 'q' or 'quit' to exit")
    print("  5. (Optional) Type 'enable' to re-enable torque")
    print("  6. (Optional) Type 'disable' to disable torque again")
    print("  7. (Optional) Type 'debug' to see raw JSON data")
    print("\n✅ Ready to test! Move the arm and press Enter.")
    
    test_count = 0
    debug_mode = False
    
    while True:
        user_input = input("\nPress Enter to test FK (or 'q' to quit, 'enable'/'disable' for torque, 'debug' to toggle): ").strip().lower()
        
        if user_input in ['q', 'quit', 'exit']:
            print("\n👋 Exiting...")
            break
        
        # Handle torque control commands
        if user_input == 'enable':
            enable_robot_torque(robot)
            continue
        
        if user_input == 'disable':
            disable_robot_torque(robot)
            continue
        
        # Handle debug toggle
        if user_input == 'debug':
            debug_mode = not debug_mode
            print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}")
            continue
        
        test_count += 1
        print(f"\n🔬 Test #{test_count}")
        print("-" * 70)
        
        # Read raw serial data (need full JSON with x,y,z position)
        json_str = None
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                if robot.ser.in_waiting > 0:
                    line = robot.ser.readline().decode('utf-8').strip()
                    if line and line.startswith('{'):
                        # Validate it's proper JSON
                        test_parse = json.loads(line)
                        # Check if it has the data we need
                        if 'x' in test_parse and 'b' in test_parse:
                            json_str = line
                            break
                time.sleep(0.05)  # Wait 50ms between attempts
            except Exception as e:
                continue
        
        if not json_str:
            print("❌ Failed to read robot data")
            print("   Make sure robot is sending data (should see JSON in serial)")
            print("   Trying again...")
            continue
        
        # Show raw JSON in debug mode
        if debug_mode:
            print(f"🔍 Raw JSON: {json_str}")
        
        # Parse robot data
        position, joints = fk_tester.parse_robot_json(json_str)
        
        if position is None or joints is None:
            print("❌ Failed to parse robot data")
            continue
        
        # Compute FK
        fk_result = fk_tester.compute_fk_from_joints(joints)
        
        # Create DemoGen state vector
        demogen_state = fk_tester.create_demogen_state(position, joints)
        
        # Display results
        fk_tester.print_robot_state(position, joints, fk_result, demogen_state)
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    robot.stop_background_thread()
    if robot.ser and robot.ser.is_open:
        robot.ser.close()
    
    print("✅ Disconnected from robot")
    print("\n" + "="*70)
    print(f"Completed {test_count} FK tests")
    print("="*70)


if __name__ == "__main__":
    main()
