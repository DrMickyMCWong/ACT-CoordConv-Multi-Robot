"""
Python wrapper for RoArm-M3 IK solver.

This module provides a Python interface to the inverse kinematics solver
for the RoArm-M3 robot arm. It handles frame transformations between the
world frame (used in recordings) and the arm frame (used by IK solver).

The IK solver is based on the C++ implementation in solver.hpp from the
roarm_ws-ros2-humble package (roarm_m3 namespace).
"""

import numpy as np
from typing import Optional, Tuple


class RoArmIKSolver:
    """
    Inverse Kinematics solver for RoArm-M3 robot arm.
    
    The solver works in the arm's local coordinate frame (origin at robot base),
    while Isaac Lab recordings are in world frame. This class handles the
    frame transformations automatically.
    
    Link lengths (from solver.hpp):
    - l2A = 236.82 mm, l2B = 30.00 mm  → l2 = sqrt(l2A² + l2B²)
    - l3A = 144.49 mm, l3B = 0 mm      → l3 = l3A
    - l4A = 171.67 mm, l4B = 13.69 mm  → lE = sqrt(l4A² + l4B²)
    """
    
    def __init__(self):
        # Link lengths from solver.hpp (roarm_m3 namespace)
        self.l2A = 236.82  # mm
        self.l2B = 30.00   # mm
        self.l2 = np.sqrt(self.l2A**2 + self.l2B**2)
        self.t2rad = np.arctan2(self.l2B, self.l2A)
        
        self.l3A = 144.49  # mm
        self.l3B = 0.0     # mm
        self.l3 = np.sqrt(self.l3A**2 + self.l3B**2)
        self.t3rad = np.arctan2(self.l3B, self.l3A)
        
        self.l4A = 171.67  # mm
        self.l4B = 13.69   # mm
        self.lEA = self.l4A  # EoAT_A = 0
        self.lEB = self.l4B  # EoAT_B = 0
        self.lE = np.sqrt(self.lEA**2 + self.lEB**2)
        self.tErad = np.arctan2(self.lEB, self.lEA)
        
        # TCP offset from l4_link (from URDF: l4_to_hand_tcp joint)
        # Origin: xyz="0 0 0.115428" rpy="1.5708 -1.5708 0"
        # This is 115.428mm in Z direction of l4_link frame
        self.tcp_offset_mm = 115.428  # mm
        
        # Robot base position in world frame (from Isaac Lab environment)
        self.robot_base_pos_world = np.array([0.15, 0.4, 0.05])  # meters
    
    def world_to_arm_frame(self, world_pos: np.ndarray) -> np.ndarray:
        """
        Convert position from world frame to arm frame.
        
        Args:
            world_pos: Position in world frame [x, y, z] in meters
            
        Returns:
            Position in arm frame [x, y, z] in millimeters
        """
        # Subtract robot base position and convert to mm
        arm_pos_m = world_pos - self.robot_base_pos_world
        arm_pos_mm = arm_pos_m * 1000.0  # Convert m to mm
        return arm_pos_mm
    
    def _cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Cartesian coordinates to polar coordinates."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    
    def _polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """Convert polar coordinates to Cartesian coordinates."""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    
    def _rotate_point(self, theta: float) -> Tuple[float, float]:
        """
        Rotate point to compensate for wrist offset.
        Corresponds to rotatePoint() in solver.hpp
        """
        alpha = self.tErad + theta
        xB = -self.lE * np.cos(alpha)
        yB = -self.lE * np.sin(alpha)
        return xB, yB
    
    def _move_point(self, xA: float, yA: float, s: float) -> Tuple[float, float]:
        """
        Move point along vector by distance s.
        Corresponds to movePoint() in solver.hpp
        """
        distance = np.sqrt(xA**2 + yA**2)
        
        if distance - s <= 1e-6:
            return 0.0, 0.0
        
        ratio = (distance - s) / distance
        xB = xA * ratio
        yB = yA * ratio
        return xB, yB
    
    def _solve_2link_ik(self, a_in: float, b_in: float) -> Tuple[float, float, float]:
        """
        Solve 2-link planar IK problem.
        Corresponds to simpleLinkageIkRad() in solver.hpp
        
        Args:
            a_in: Horizontal distance in mm
            b_in: Vertical distance in mm
            
        Returns:
            (shoulder_rad, elbow_rad, eoat_buffer_rad)
        """
        LA = self.l2
        LB = self.l3
        
        if abs(b_in) < 1e-6:
            # Vertical case (b_in ≈ 0)
            psi = np.arccos((LA**2 + a_in**2 - LB**2) / (2 * LA * a_in)) + self.t2rad
            alpha = np.pi / 2.0 - psi
            omega = np.arccos((a_in**2 + LB**2 - LA**2) / (2 * a_in * LB))
            beta = psi + omega - self.t3rad
        else:
            # General case
            L2C = a_in**2 + b_in**2
            LC = np.sqrt(L2C)
            lambda_angle = np.arctan2(b_in, a_in)
            psi = np.arccos((LA**2 + L2C - LB**2) / (2 * LA * LC)) + self.t2rad
            alpha = np.pi / 2.0 - lambda_angle - psi
            omega = np.arccos((LB**2 + L2C - LA**2) / (2 * LC * LB))
            beta = psi + omega - self.t3rad
        
        delta = np.pi / 2.0 - alpha - beta
        
        # Check for NaN
        if np.isnan(alpha) or np.isnan(beta) or np.isnan(delta):
            return None, None, None
        
        return alpha, beta, delta
    
    def solve_ik(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None,
                 target_pitch: Optional[float] = None, debug: bool = False) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target end-effector (hand_tcp) pose.
        
        Uses empirical approach: subtracts estimated TCP offset to get l4_link position,
        then solves 2-link planar IK.
        
        Args:
            target_pos: Target TCP position in world frame [x, y, z] in meters
            target_quat: Target quaternion [x, y, z, w] (optional)
            target_pitch: Target gripper pitch in radians (optional, overrides quat)
            debug: Print debug information
            
        Returns:
            Joint angles [base, shoulder, elbow, wrist_pitch, wrist_roll] in radians,
            or None if IK fails
        """
        # Convert world frame to arm frame (mm)
        tcp_pos_mm = self.world_to_arm_frame(target_pos)
        
        if debug:
            print(f"  [IK Debug] World pos (TCP): {target_pos}")
            print(f"  [IK Debug] Arm pos (TCP, mm): {tcp_pos_mm}")
        
        # Estimate gripper pitch from quaternion
        if target_pitch is None and target_quat is not None:
            # Convert quaternion to pitch
            # Simplified: extract pitch from quaternion [x, y, z, w]
            x, y, z, w = target_quat
            sinp = 2.0 * (w * y - z * x)
            pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
            
            if debug:
                print(f"  [IK Debug] Quat pitch: {np.rad2deg(pitch):.1f}°")
        elif target_pitch is not None:
            pitch = target_pitch
        else:
            # Default: estimate from height (lower Z → more downward)
            pitch = -np.pi / 3.0  # -60° default
        
        # Convert pitch to gripper angle in joint space
        # Empirically: joint gripper angle ≈ 90° + pitch_deg
        # (This is approximate - will be refined by IK solution)
        estimated_gripper_angle = np.pi / 2.0 - pitch
        
        # Estimate TCP offset based on gripper angle
        # From empirical data: offset magnitude varies 26-48mm
        # Use simple model: offset is mostly along gripper direction, ~40mm average
        tcp_offset_mag = 40.0  # mm (empirical average)
        
        tcp_offset_r, tcp_offset_z = self._polar_to_cartesian(
            tcp_offset_mag,
            (np.pi / 2.0) - estimated_gripper_angle
        )
        
        # Get base angle from XY position
        r_tcp, theta_base = self._cartesian_to_polar(tcp_pos_mm[0], tcp_pos_mm[1])
        
        # Subtract TCP offset to estimate l4_link position
        r_l4 = r_tcp - tcp_offset_r
        z_l4 = tcp_pos_mm[2] - tcp_offset_z
        
        if debug:
            print(f"  [IK Debug] Estimated gripper angle: {np.rad2deg(estimated_gripper_angle):.1f}°")
            print(f"  [IK Debug] TCP offset estimate: r={tcp_offset_r:.1f}, z={tcp_offset_z:.1f} mm")
            print(f"  [IK Debug] l4_link pos (est): r={r_l4:.1f}, z={z_l4:.1f} mm")
        
        # Solve 2-link planar IK for l4_link position
        shoulder_rad, elbow_rad, eoat_buffer = self._solve_2link_ik(r_l4, z_l4)
        
        # Check if IK failed
        if shoulder_rad is None:
            if debug:
                print(f"  [IK Debug] ❌ 2-link IK failed")
                print(f"  [IK Debug]    Reach: l2+l3={self.l2 + self.l3:.1f}mm")
                print(f"  [IK Debug]    Distance: {np.sqrt(r_l4**2 + z_l4**2):.1f}mm")
            return None
        
        # Calculate wrist pitch to achieve target gripper angle
        # Target: shoulder + elbow + wrist_pitch = estimated_gripper_angle
        wrist_pitch_rad = estimated_gripper_angle - shoulder_rad - elbow_rad
        
        # Default roll
        roll = 0.0
        
        if debug:
            print(f"  [IK Debug] 2-link IK: shoulder={np.rad2deg(shoulder_rad):.1f}°, elbow={np.rad2deg(elbow_rad):.1f}°")
            print(f"  [IK Debug] Wrist pitch: {np.rad2deg(wrist_pitch_rad):.1f}°")
            actual_gripper = shoulder_rad + elbow_rad + wrist_pitch_rad
            print(f"  [IK Debug] Actual gripper angle: {np.rad2deg(actual_gripper):.1f}°")
        
        # Assemble joint angles
        joint_angles = np.array([
            theta_base,
            shoulder_rad,
            elbow_rad,
            wrist_pitch_rad,
            roll
        ])
        
        if debug:
            print(f"  [IK Debug] ✅ Solution (deg): [{', '.join(f'{np.rad2deg(a):.1f}' for a in joint_angles)}]")
        
        return joint_angles
    
    def solve_fk(self, joint_angles: np.ndarray, include_tcp_offset: bool = True) -> np.ndarray:
        """
        Solve forward kinematics for given joint angles.
        
        Computes position of hand_tcp (gripper TCP) from joint angles.
        
        Args:
            joint_angles: Joint angles [base, shoulder, elbow, wrist_pitch, wrist_roll]
                         in radians
            include_tcp_offset: If True, add TCP offset (115.428mm along gripper).
                              If False, return l4_link end position.
            
        Returns:
            End-effector position in arm frame [x, y, z] in millimeters
        """
        base_rad, shoulder_rad, elbow_rad, wrist_pitch_rad, roll_rad = joint_angles
        
        # Calculate cumulative link positions to get l4_link + lE position
        # This follows the C++ FK: sum of l2, l3, and lE contributions
        
        # Link 2 contribution
        r1, z1 = self._polar_to_cartesian(
            self.l2,
            (np.pi / 2.0) - (shoulder_rad + self.t2rad)
        )
        
        # Link 3 contribution  
        r2, z2 = self._polar_to_cartesian(
            self.l3,
            (np.pi / 2.0) - (elbow_rad + shoulder_rad + self.t3rad)
        )
        
        # Link 4 + lE contribution (to end of l4_link)
        r3, z3 = self._polar_to_cartesian(
            self.lE,
            (np.pi / 2.0) - (elbow_rad + shoulder_rad + wrist_pitch_rad + self.tErad)
        )
        
        # Position at l4_link end (in arm's radial-z plane)
        r_l4 = r1 + r2 + r3
        z_l4 = z1 + z2 + z3
        
        # Add TCP offset if requested
        # TCP is 115.428mm further along the gripper direction
        # Gripper direction angle = shoulder + elbow + wrist_pitch
        if include_tcp_offset:
            gripper_angle_rad = shoulder_rad + elbow_rad + wrist_pitch_rad
            
            # TCP offset vector in radial-z plane
            tcp_r, tcp_z = self._polar_to_cartesian(
                self.tcp_offset_mm,
                (np.pi / 2.0) - gripper_angle_rad
            )
            
            r_ee = r_l4 + tcp_r
            z_ee = z_l4 + tcp_z
        else:
            r_ee = r_l4
            z_ee = z_l4
        
        # Convert from arm's radial-z to world XY
        x_ee, y_ee = self._polar_to_cartesian(r_ee, base_rad)
        
        return np.array([x_ee, y_ee, z_ee])
    
    def fk_to_world_frame(self, arm_pos_mm: np.ndarray) -> np.ndarray:
        """
        Convert position from arm frame to world frame.
        
        Args:
            arm_pos_mm: Position in arm frame [x, y, z] in millimeters
            
        Returns:
            Position in world frame [x, y, z] in meters
        """
        # Convert mm to m
        arm_pos_m = arm_pos_mm / 1000.0
        # Add robot base position
        world_pos = arm_pos_m + self.robot_base_pos_world
        return world_pos


if __name__ == "__main__":
    # Simple test
    solver = RoArmIKSolver()
    
    print("RoArm-M3 IK Solver Test")
    print("=" * 50)
    print(f"Link lengths: l2={solver.l2:.2f}mm, l3={solver.l3:.2f}mm, lE={solver.lE:.2f}mm")
    print(f"Robot base (world): {solver.robot_base_pos_world}")
    
    # Test FK → IK → FK cycle
    test_joints = np.array([0.0, 0.5, 1.5, 0.3, 0.0])  # radians
    
    print(f"\nTest joint angles: {test_joints}")
    
    # FK
    ee_pos_arm = solver.solve_fk(test_joints)
    ee_pos_world = solver.fk_to_world_frame(ee_pos_arm)
    print(f"FK result (arm frame): {ee_pos_arm} mm")
    print(f"FK result (world frame): {ee_pos_world} m")
    
    # IK
    recovered_joints = solver.solve_ik(ee_pos_world)
    print(f"IK result: {recovered_joints}")
    
    # Verify
    if recovered_joints is not None:
        error = np.abs(recovered_joints - test_joints)
        print(f"Joint angle errors: {np.rad2deg(error)} degrees")
