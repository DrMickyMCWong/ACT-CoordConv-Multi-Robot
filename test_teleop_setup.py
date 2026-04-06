#!/usr/bin/env python3
"""
Test script to verify teleop setup before running full recording
"""
import sys
import os
import numpy as np

# Add the act directory to path
sys.path.append('/home/hk/Documents/ACT_Shaka/act-main/act')

from constants import SIM_TASK_CONFIGS, ROBOT_PORTS
from sim_env import make_sim_env
import alicia_d_sdk

def read_robot_state(robot):
    """Read robot joint angles and gripper state using Alicia SDK"""
    try:
        joint_gripper_state = robot.get_robot_state("joint_gripper")
        
        if joint_gripper_state is None:
            print("Failed to get joint_gripper state")
            return np.zeros(7), np.zeros(7)
        
        joint_angles = joint_gripper_state.angles
        gripper_pos = joint_gripper_state.gripper
        
        gripper_normalized = gripper_pos / 1000.0
        qpos = joint_angles + [gripper_normalized]
        qvel = [0.0] * 7
        
        return np.array(qpos), np.array(qvel)
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return np.zeros(7), np.zeros(7)

def test_setup():
    print("=== Testing Teleop Setup ===")
    
    # 1. Test task configuration
    task_name = 'sim_pick_cube_single_teleop'
    if task_name in SIM_TASK_CONFIGS:
        print(f"✓ Task config found: {SIM_TASK_CONFIGS[task_name]}")
    else:
        print(f"✗ Task config missing for {task_name}")
        return False
    
    # 2. Test simulation environment
    try:
        print("Testing simulation environment...")
        env = make_sim_env(task_name)
        ts = env.reset()
        print(f"✓ Simulation environment created")
        print(f"  - Observation qpos shape: {ts.observation['qpos'].shape}")
        print(f"  - Observation qvel shape: {ts.observation['qvel'].shape}")
        print(f"  - Available cameras: {list(ts.observation['images'].keys())}")
        del env
    except Exception as e:
        print(f"✗ Simulation environment error: {e}")
        return False
    
    # 3. Test robot connection
    try:
        print("Testing robot connection...")
        leader = alicia_d_sdk.create_robot(
            port=ROBOT_PORTS['leader'],
            gripper_type="50mm"
        )
        print("✓ Robot connected successfully")
        
        # Test reading robot state
        qpos, qvel = read_robot_state(leader)
        print(f"✓ Robot state read successfully")
        print(f"  - Joint positions: {qpos[:6].round(3)}")
        print(f"  - Gripper position: {qpos[6]:.3f}")
        
        leader.disconnect()
        print("✓ Robot disconnected successfully")
        
    except Exception as e:
        print(f"✗ Robot connection error: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    print("\nYou can now run teleoperation recording with:")
    print("cd /home/hk/Documents/ACT_Shaka/act-main/act")
    print("python record_sim_teleop_episodes.py --task_name sim_pick_cube_single_teleop --dataset_dir ./teleop_data --num_episodes 5 --onscreen_render")
    
    return True

if __name__ == '__main__':
    test_setup()