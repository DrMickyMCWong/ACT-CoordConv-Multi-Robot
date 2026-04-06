#!/usr/bin/env python3
"""
Export Episode Data from HDF5 to separate files

This script exports recorded episode data to:
1. RGB images (PNG) - extracted from RGBD channels 0-2
2. Depth images (PNG) - extracted from RGBD channel 3, colorized
3. Depth raw data (NPY) - raw depth values for analysis
4. Joint states (NPY + CSV) - follower (observation) and leader (action) poses
5. End-effector positions (NPY + CSV) - robot frame x, y, z coordinates

Usage:
    python export_episode_data.py --episode_path data/task7/episode_0.hdf5
    python export_episode_data.py --episode_path data/task7/episode_0.hdf5 --output_dir exports/task7_ep0
"""

import h5py
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import csv
from tqdm import tqdm


def export_episode(episode_path: str, output_dir: str = None):
    """
    Export all data from an episode HDF5 file.
    
    Args:
        episode_path: Path to the HDF5 episode file
        output_dir: Output directory (default: auto-generated from episode path)
    """
    
    # Create output directory
    if output_dir is None:
        episode_file = Path(episode_path)
        output_dir = episode_file.parent / f"{episode_file.stem}_exported"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"Exporting Episode Data")
    print("="*70)
    print(f"Source: {episode_path}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Create subdirectories
    rgb_dir = output_dir / "rgb_images"
    depth_colormap_dir = output_dir / "depth_colormap_images"
    depth_png_dir = output_dir / "depth_png"
    depth_raw_dir = output_dir / "depth_raw"
    joint_states_dir = output_dir / "joint_states"
    follower_poses_dir = output_dir / "follower_poses_per_frame"
    leader_poses_dir = output_dir / "leader_poses_per_frame"
    follower_qvel_dir = output_dir / "follower_qvel_per_frame"
    ee_pos_dir = output_dir / "ee_positions"
    
    for dir_path in [rgb_dir, depth_colormap_dir, depth_png_dir, depth_raw_dir, 
                     joint_states_dir, follower_poses_dir, leader_poses_dir, follower_qvel_dir, ee_pos_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load HDF5 file
    print("\n📂 Loading HDF5 file...")
    with h5py.File(episode_path, 'r') as f:
        # Get data info
        print("\n📊 Dataset Structure:")
        print(f"  Observations:")
        for key in f['observations'].keys():
            if key == 'images':
                for cam_name in f['observations/images'].keys():
                    shape = f[f'observations/images/{cam_name}'].shape
                    dtype = f[f'observations/images/{cam_name}'].dtype
                    print(f"    images/{cam_name}: {shape}, {dtype}")
            else:
                shape = f[f'observations/{key}'].shape
                dtype = f[f'observations/{key}'].dtype
                print(f"    {key}: {shape}, {dtype}")
        
        action_shape = f['action'].shape
        action_dtype = f['action'].dtype
        print(f"  Actions: {action_shape}, {action_dtype}")
        
        # Load all data
        print("\n📥 Loading data into memory...")
        qpos = f['observations/qpos'][:]  # Follower joint states
        qvel = f['observations/qvel'][:]  # Follower joint velocities
        
        # Check if ee_pos exists
        has_ee_pos = 'ee_pos' in f['observations']
        if has_ee_pos:
            ee_pos = f['observations/ee_pos'][:]  # End-effector positions
            print(f"  ✓ Found end-effector positions: {ee_pos.shape}")
        else:
            print(f"  ⚠ No end-effector positions found in this episode")
        
        action = f['action'][:]  # Leader joint states (target)
        
        # Get images (assumes 'front' camera)
        camera_names = list(f['observations/images'].keys())
        print(f"  Cameras: {camera_names}")
        
        images_rgbd = f['observations/images/front'][:]  # (T, H, W, 4)
        
        num_frames = images_rgbd.shape[0]
        height, width = images_rgbd.shape[1:3]
        num_channels = images_rgbd.shape[3]
        
        print(f"\n📹 Video Info:")
        print(f"  Total frames: {num_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Channels: {num_channels} ({'RGBD' if num_channels == 4 else 'RGB'})")
        
    # Export images
    print(f"\n🖼️  Exporting images...")
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        rgbd_frame = images_rgbd[frame_idx]
        
        if num_channels == 4:
            # Split RGBD into RGB + D
            rgb_frame = rgbd_frame[:, :, :3]  # First 3 channels (RGB)
            depth_frame = rgbd_frame[:, :, 3]  # 4th channel (normalized depth 0-255)
        else:
            # Only RGB available
            rgb_frame = rgbd_frame
            depth_frame = None
        
        # Export RGB image
        rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        rgb_filename = rgb_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(rgb_filename), rgb_bgr)
        
        # Export depth if available
        if depth_frame is not None:
            # Save depth as grayscale PNG (for easy viewing)
            depth_png_filename = depth_png_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(depth_png_filename), depth_frame)
            
            # Save colorized depth (for visualization)
            depth_colormap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            depth_colormap_filename = depth_colormap_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(depth_colormap_filename), depth_colormap)
            
            # Save raw depth values (for analysis)
            depth_raw_filename = depth_raw_dir / f"frame_{frame_idx:06d}.npy"
            np.save(str(depth_raw_filename), depth_frame)
        
        # Export individual frame poses
        follower_pose_filename = follower_poses_dir / f"frame_{frame_idx:06d}.npy"
        np.save(str(follower_pose_filename), qpos[frame_idx])
        
    # Export follower velocities per-frame
    follower_qvel_filename = follower_qvel_dir / f"frame_{frame_idx:06d}.npy"
    np.save(str(follower_qvel_filename), qvel[frame_idx])

    leader_pose_filename = leader_poses_dir / f"frame_{frame_idx:06d}.npy"
    np.save(str(leader_pose_filename), action[frame_idx])
    
    print(f"  ✓ Exported {num_frames} RGB images to: {rgb_dir}")
    if num_channels == 4:
        print(f"  ✓ Exported {num_frames} depth PNG to: {depth_png_dir}")
        print(f"  ✓ Exported {num_frames} depth colormaps to: {depth_colormap_dir}")
        print(f"  ✓ Exported {num_frames} raw depth files to: {depth_raw_dir}")
    print(f"  ✓ Exported {num_frames} follower poses to: {follower_poses_dir}")
    print(f"  ✓ Exported {num_frames} leader poses to: {leader_poses_dir}")
    
    # Export joint states
    print(f"\n🤖 Exporting joint states...")
    
    # Follower (observation) - qpos
    follower_qpos_npy = joint_states_dir / "follower_qpos.npy"
    np.save(str(follower_qpos_npy), qpos)
    print(f"  ✓ Saved follower qpos (NPY): {follower_qpos_npy}")
    
    # Follower qpos CSV
    follower_qpos_csv = joint_states_dir / "follower_qpos.csv"
    with open(follower_qpos_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'base', 'shoulder', 'elbow', 'wrist', 'roll', 'gripper'])
        for i, joints in enumerate(qpos):
            writer.writerow([i] + joints.tolist())
    print(f"  ✓ Saved follower qpos (CSV): {follower_qpos_csv}")
    
    # Follower (observation) - qvel
    follower_qvel_npy = joint_states_dir / "follower_qvel.npy"
    np.save(str(follower_qvel_npy), qvel)
    print(f"  ✓ Saved follower qvel (NPY): {follower_qvel_npy}")
    
    follower_qvel_csv = joint_states_dir / "follower_qvel.csv"
    with open(follower_qvel_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'base_vel', 'shoulder_vel', 'elbow_vel', 'wrist_vel', 'roll_vel', 'gripper_vel'])
        for i, vels in enumerate(qvel):
            writer.writerow([i] + vels.tolist())
    print(f"  ✓ Saved follower qvel (CSV): {follower_qvel_csv}")
    
    # Leader (action)
    leader_action_npy = joint_states_dir / "leader_action.npy"
    np.save(str(leader_action_npy), action)
    print(f"  ✓ Saved leader action (NPY): {leader_action_npy}")
    
    leader_action_csv = joint_states_dir / "leader_action.csv"
    with open(leader_action_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'base', 'shoulder', 'elbow', 'wrist', 'roll', 'gripper'])
        for i, joints in enumerate(action):
            writer.writerow([i] + joints.tolist())
    print(f"  ✓ Saved leader action (CSV): {leader_action_csv}")
    
    # Export end-effector positions if available
    if has_ee_pos:
        print(f"\n📍 Exporting end-effector positions...")
        
        ee_pos_npy = ee_pos_dir / "ee_positions.npy"
        np.save(str(ee_pos_npy), ee_pos)
        print(f"  ✓ Saved EE positions (NPY): {ee_pos_npy}")
        
        ee_pos_csv = ee_pos_dir / "ee_positions.csv"
        with open(ee_pos_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'x_mm', 'y_mm', 'z_mm'])
            for i, pos in enumerate(ee_pos):
                writer.writerow([i] + pos.tolist())
        print(f"  ✓ Saved EE positions (CSV): {ee_pos_csv}")
    
    # Create summary file
    print(f"\n📄 Creating summary...")
    summary_file = output_dir / "export_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Episode Export Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Source File: {episode_path}\n")
        f.write(f"Export Date: {Path(episode_path).stat().st_mtime}\n")
        f.write(f"\nData Information:\n")
        f.write(f"  Total Frames: {num_frames}\n")
        f.write(f"  Image Resolution: {width}x{height}\n")
        f.write(f"  Image Channels: {num_channels} ({'RGBD' if num_channels == 4 else 'RGB'})\n")
        f.write(f"  Joint State Dimension: {qpos.shape[1]}\n")
        f.write(f"  Has End-Effector Positions: {has_ee_pos}\n")
        f.write(f"\nExported Files:\n")
        f.write(f"  RGB Images: {rgb_dir.name}/ ({num_frames} files)\n")
        if num_channels == 4:
            f.write(f"  Depth PNG (grayscale): {depth_png_dir.name}/ ({num_frames} files)\n")
            f.write(f"  Depth Colormaps: {depth_colormap_dir.name}/ ({num_frames} files)\n")
            f.write(f"  Depth Raw Data: {depth_raw_dir.name}/ ({num_frames} files)\n")
        f.write(f"  Follower Poses (per frame): {follower_poses_dir.name}/ ({num_frames} files)\n")
        f.write(f"  Leader Poses (per frame): {leader_poses_dir.name}/ ({num_frames} files)\n")
        f.write(f"  Joint States: {joint_states_dir.name}/\n")
        f.write(f"    - follower_qpos.npy, follower_qpos.csv\n")
        f.write(f"    - follower_qvel.npy, follower_qvel.csv\n")
        f.write(f"    - leader_action.npy, leader_action.csv\n")
        if has_ee_pos:
            f.write(f"  EE Positions: {ee_pos_dir.name}/\n")
            f.write(f"    - ee_positions.npy, ee_positions.csv\n")
        f.write(f"\nJoint Order (indices 0-5):\n")
        f.write(f"  0: Base (radians)\n")
        f.write(f"  1: Shoulder (radians)\n")
        f.write(f"  2: Elbow (radians)\n")
        f.write(f"  3: Wrist (radians)\n")
        f.write(f"  4: Roll (radians)\n")
        f.write(f"  5: Gripper (radians)\n")
        if has_ee_pos:
            f.write(f"\nEnd-Effector Position Format:\n")
            f.write(f"  x: millimeters (forward/backward)\n")
            f.write(f"  y: millimeters (left/right)\n")
            f.write(f"  z: millimeters (up/down)\n")
        f.write(f"\nDepth Information (if RGBD):\n")
        f.write(f"  Normalized range: 0-255 (uint8)\n")
        f.write(f"  Actual range: 200mm - 1500mm\n")
        f.write(f"  Formula: depth_mm ≈ 200 + (normalized_value / 255) * 1300\n")
        f.write(f"\nStatistics:\n")
        f.write(f"  Follower qpos range:\n")
        for i in range(qpos.shape[1]):
            f.write(f"    Joint {i}: [{qpos[:, i].min():.4f}, {qpos[:, i].max():.4f}] rad\n")
        f.write(f"  Leader action range:\n")
        for i in range(action.shape[1]):
            f.write(f"    Joint {i}: [{action[:, i].min():.4f}, {action[:, i].max():.4f}] rad\n")
        if has_ee_pos:
            f.write(f"  EE position range:\n")
            f.write(f"    X: [{ee_pos[:, 0].min():.2f}, {ee_pos[:, 0].max():.2f}] mm\n")
            f.write(f"    Y: [{ee_pos[:, 1].min():.2f}, {ee_pos[:, 1].max():.2f}] mm\n")
            f.write(f"    Z: [{ee_pos[:, 2].min():.2f}, {ee_pos[:, 2].max():.2f}] mm\n")
    
    print(f"  ✓ Saved summary: {summary_file}")
    
    # Print statistics
    print(f"\n📈 Statistics:")
    print(f"  Follower joint ranges (radians):")
    for i in range(qpos.shape[1]):
        print(f"    Joint {i}: [{qpos[:, i].min():7.4f}, {qpos[:, i].max():7.4f}]")
    print(f"  Leader joint ranges (radians):")
    for i in range(action.shape[1]):
        print(f"    Joint {i}: [{action[:, i].min():7.4f}, {action[:, i].max():7.4f}]")
    
    if has_ee_pos:
        print(f"  End-effector position ranges (mm):")
        print(f"    X: [{ee_pos[:, 0].min():7.2f}, {ee_pos[:, 0].max():7.2f}]")
        print(f"    Y: [{ee_pos[:, 1].min():7.2f}, {ee_pos[:, 1].max():7.2f}]")
        print(f"    Z: [{ee_pos[:, 2].min():7.2f}, {ee_pos[:, 2].max():7.2f}]")
    
    print("\n" + "="*70)
    print("✅ Export Complete!")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nFolder structure:")
    print(f"  {output_dir.name}/")
    print(f"  ├── rgb_images/          ({num_frames} PNG files)")
    if num_channels == 4:
        print(f"  ├── depth_colormap_images/ ({num_frames} PNG files)")
        print(f"  ├── depth_raw/           ({num_frames} NPY files)")
    print(f"  ├── joint_states/        (6 files: NPY + CSV)")
    if has_ee_pos:
        print(f"  ├── ee_positions/        (2 files: NPY + CSV)")
    print(f"  └── export_summary.txt")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Export episode data from HDF5 to separate files')
    parser.add_argument('--episode_path', type=str, required=True,
                       help='Path to the episode HDF5 file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated from episode path)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.episode_path):
        print(f"❌ Error: File not found: {args.episode_path}")
        return
    
    # Export
    export_episode(args.episode_path, args.output_dir)


if __name__ == '__main__':
    main()
