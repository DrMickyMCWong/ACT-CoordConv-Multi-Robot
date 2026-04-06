#!/usr/bin/env python3
"""
Test script to verify IK wrapper works with recorded seed demonstrations.

This script:
1. Loads waypoint data from seed_demos.hdf5
2. Tests Forward Kinematics: joint_angles → EE position
3. Tests Inverse Kinematics: EE position → joint_angles
4. Verifies frame transformations (world ↔ arm frame)
5. Reports errors and validates IK accuracy
"""

import h5py
import numpy as np
from roarm_ik_wrapper import RoArmIKSolver


def test_forward_kinematics(ik_solver, demo_name, waypoint_data):
    """Test FK: recorded joint angles should produce recorded EE positions"""
    print(f"\n{'='*70}")
    print(f"Testing Forward Kinematics - {demo_name}")
    print(f"{'='*70}")
    
    errors = []
    max_error = 0
    max_error_waypoint = None
    
    for wp_name, wp_group in waypoint_data.items():
        joint_pos = wp_group['joint_pos'][:]
        recorded_ee_pos = wp_group['ee_pos'][:]
        
        # Run FK: joint angles → EE position (returns in arm frame, mm)
        computed_ee_pos_arm = ik_solver.solve_fk(joint_pos)
        
        # Convert arm frame to world frame for comparison
        computed_ee_pos_world = ik_solver.fk_to_world_frame(computed_ee_pos_arm)
        error = np.linalg.norm(computed_ee_pos_world - recorded_ee_pos)
        errors.append(error)
        
        if error > max_error:
            max_error = error
            max_error_waypoint = wp_name
        
        # Print details
        print(f"\n  {wp_name}:")
        print(f"    Joint angles: {joint_pos}")
        print(f"    Recorded EE:  {recorded_ee_pos}")
        print(f"    Computed EE:  {computed_ee_pos_world}")
        print(f"    Error: {error*1000:.2f} mm")
    
    avg_error = np.mean(errors)
    print(f"\n  📊 FK Statistics:")
    print(f"     Average error: {avg_error*1000:.2f} mm")
    print(f"     Max error: {max_error*1000:.2f} mm (at {max_error_waypoint})")
    print(f"     Min error: {np.min(errors)*1000:.2f} mm")
    
    # Validate FK accuracy (should be < 5mm error)
    if avg_error < 0.005:  # 5mm
        print(f"  ✅ FK PASSED: Average error < 5mm")
        return True
    else:
        print(f"  ❌ FK FAILED: Average error > 5mm")
        return False


def test_inverse_kinematics(ik_solver, demo_name, waypoint_data):
    """Test IK: recorded EE positions should produce similar joint angles"""
    print(f"\n{'='*70}")
    print(f"Testing Inverse Kinematics - {demo_name}")
    print(f"{'='*70}")
    
    position_errors = []
    joint_errors = []
    ik_failures = 0
    
    for wp_name, wp_group in waypoint_data.items():
        recorded_joint_pos = wp_group['joint_pos'][:]
        recorded_ee_pos = wp_group['ee_pos'][:]
        recorded_ee_quat = wp_group['ee_quat'][:]
        
        # Enable debug for first waypoint only
        debug = (wp_name == 'waypoint_0')
        
        # Run IK: EE position → joint angles
        computed_joint_pos = ik_solver.solve_ik(
            target_pos=recorded_ee_pos,
            target_quat=recorded_ee_quat,
            debug=debug
        )
        
        if computed_joint_pos is None:
            print(f"\n  {wp_name}:")
            print(f"    ❌ IK FAILED: No solution found")
            print(f"    Target EE: {recorded_ee_pos}")
            ik_failures += 1
            continue
        
        # Verify IK result by running FK (returns in arm frame)
        verified_ee_pos_arm = ik_solver.solve_fk(computed_joint_pos)
        # Convert to world frame for comparison
        verified_ee_pos_world = ik_solver.fk_to_world_frame(verified_ee_pos_arm)
        
        # Calculate position error (how close did we get to target?)
        pos_error = np.linalg.norm(verified_ee_pos_world - recorded_ee_pos)
        position_errors.append(pos_error)
        
        # Calculate joint angle error (how different are the joint angles?)
        joint_diff = np.abs(np.array(computed_joint_pos) - recorded_joint_pos)
        joint_error = np.mean(joint_diff)
        joint_errors.append(joint_error)
        
        # Print details
        print(f"\n  {wp_name}:")
        print(f"    Target EE:       {recorded_ee_pos}")
        print(f"    Recorded joints: {recorded_joint_pos}")
        print(f"    Computed joints: {computed_joint_pos}")
        print(f"    Verified EE:     {verified_ee_pos_world}")
        print(f"    Position error:  {pos_error*1000:.2f} mm")
        print(f"    Joint angle error: {np.rad2deg(joint_error):.2f} deg (avg)")
        print(f"    Joint diffs (deg): {np.rad2deg(joint_diff)}")
    
    # Print statistics
    if len(position_errors) > 0:
        avg_pos_error = np.mean(position_errors)
        avg_joint_error = np.mean(joint_errors)
        
        print(f"\n  📊 IK Statistics:")
        print(f"     Successful IKs: {len(position_errors)}/{len(waypoint_data)}")
        print(f"     Failed IKs: {ik_failures}")
        print(f"     Average position error: {avg_pos_error*1000:.2f} mm")
        print(f"     Max position error: {np.max(position_errors)*1000:.2f} mm")
        print(f"     Average joint error: {np.rad2deg(avg_joint_error):.2f} deg")
        
        # Validate IK accuracy
        success = True
        if avg_pos_error < 0.005:  # 5mm
            print(f"  ✅ IK Position Accuracy PASSED: < 5mm")
        else:
            print(f"  ❌ IK Position Accuracy FAILED: > 5mm")
            success = False
        
        if ik_failures == 0:
            print(f"  ✅ IK Reliability PASSED: No failures")
        else:
            print(f"  ⚠️  IK Reliability WARNING: {ik_failures} failures")
            success = False
        
        return success
    else:
        print(f"  ❌ IK FAILED: All solutions failed")
        return False


def test_frame_transformations(ik_solver):
    """Test frame transformations between world and arm frames"""
    print(f"\n{'='*70}")
    print(f"Testing Frame Transformations")
    print(f"{'='*70}")
    
    # Test case 1: Robot base position should map to (0, 0, 0) in arm frame
    robot_base_world = np.array([0.15, 0.4, 0.05])  # meters
    arm_frame_pos = ik_solver.world_to_arm_frame(robot_base_world)
    
    print(f"\n  Test 1: Robot base position")
    print(f"    World frame: {robot_base_world}")
    print(f"    Arm frame:   {arm_frame_pos}")
    
    expected = np.array([0, 0, 0])
    error = np.linalg.norm(arm_frame_pos - expected)
    
    if error < 1.0:  # 1mm tolerance
        print(f"    ✅ PASSED: Error {error:.2f} mm")
        test1_pass = True
    else:
        print(f"    ❌ FAILED: Error {error:.2f} mm")
        test1_pass = False
    
    # Test case 2: Known world position → arm frame → verify
    test_world_pos = np.array([0.25, 0.45, 0.10])  # meters
    test_arm_pos = ik_solver.world_to_arm_frame(test_world_pos)
    
    # Expected: subtract base, convert to mm
    expected_arm = (test_world_pos - robot_base_world) * 1000  # mm
    error = np.linalg.norm(test_arm_pos - expected_arm)
    
    print(f"\n  Test 2: Arbitrary world position")
    print(f"    World frame: {test_world_pos}")
    print(f"    Arm frame:   {test_arm_pos}")
    print(f"    Expected:    {expected_arm}")
    
    if error < 1.0:  # 1mm tolerance
        print(f"    ✅ PASSED: Error {error:.2f} mm")
        test2_pass = True
    else:
        print(f"    ❌ FAILED: Error {error:.2f} mm")
        test2_pass = False
    
    return test1_pass and test2_pass


def main():
    print("="*70)
    print("RoArm IK Wrapper Verification Test")
    print("="*70)
    
    # Initialize IK solver
    print("\n📐 Initializing IK solver...")
    ik_solver = RoArmIKSolver()
    print(f"   Robot base position (world frame): {ik_solver.robot_base_pos_world}")
    print(f"   Link lengths (mm): l2A={ik_solver.l2A:.2f}, l3A={ik_solver.l3A:.2f}, l4A={ik_solver.l4A:.2f}")
    
    # Test frame transformations first
    frame_test_passed = test_frame_transformations(ik_solver)
    
    # Load recorded demonstrations
    hdf5_path = "data/seed_demos.hdf5"
    print(f"\n📂 Loading demonstrations from: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Navigate to data group
            data_group = f['data']
            print(f"   Found {len(data_group.keys())} demonstrations")
            
            # Test first 3 demos (to keep output manageable)
            demos_to_test = list(data_group.keys())[:3]
            
            fk_results = []
            ik_results = []
            
            for demo_name in demos_to_test:
                demo_group = data_group[demo_name]
                waypoints_group = demo_group['waypoints']
                
                print(f"\n{'='*70}")
                print(f"Demo: {demo_name} ({len(waypoints_group.keys())} waypoints)")
                print(f"{'='*70}")
                
                # Test FK
                fk_passed = test_forward_kinematics(ik_solver, demo_name, waypoints_group)
                fk_results.append(fk_passed)
                
                # Test IK
                ik_passed = test_inverse_kinematics(ik_solver, demo_name, waypoints_group)
                ik_results.append(ik_passed)
            
            # Print final summary
            print(f"\n{'='*70}")
            print(f"FINAL TEST SUMMARY")
            print(f"{'='*70}")
            print(f"\n  Frame Transformation Tests: {'✅ PASSED' if frame_test_passed else '❌ FAILED'}")
            print(f"\n  Forward Kinematics Tests:")
            for i, (demo_name, passed) in enumerate(zip(demos_to_test, fk_results)):
                print(f"    {demo_name}: {'✅ PASSED' if passed else '❌ FAILED'}")
            
            print(f"\n  Inverse Kinematics Tests:")
            for i, (demo_name, passed) in enumerate(zip(demos_to_test, ik_results)):
                print(f"    {demo_name}: {'✅ PASSED' if passed else '❌ FAILED'}")
            
            # Overall result
            all_passed = frame_test_passed and all(fk_results) and all(ik_results)
            
            print(f"\n{'='*70}")
            if all_passed:
                print(f"✅ ALL TESTS PASSED - IK Wrapper is working correctly!")
                print(f"   You can proceed to implement trajectory generation.")
            else:
                print(f"❌ SOME TESTS FAILED - IK Wrapper needs debugging.")
                print(f"   Review the errors above before proceeding.")
            print(f"{'='*70}\n")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find {hdf5_path}")
        print(f"   Make sure you've recorded seed demonstrations first.")
    except Exception as e:
        print(f"❌ Error loading demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
