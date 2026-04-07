#!/usr/bin/env python3
"""
Post-process recorded episodes to generate point clouds from depth data.

This script reads existing HDF5 episodes and adds point cloud data to them.
Run this after recording episodes to enable point cloud visualization.
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path

# Import the same functions used during recording
import sys
sys.path.append('/home/hk/Documents/ACT_Shaka/act-main/act')

from constants import SIM_TASK_CONFIGS
from sim_env import make_sim_env

def get_camera_intrinsics(camera_name, height=480, width=640, physics=None):
    """Get camera intrinsic parameters for MuJoCo camera"""
    # Default field of view - can be read from physics if available
    if physics is not None:
        try:
            cam_id = physics.model.name2id(camera_name, 'camera')
            fovy_rad = physics.model.cam_fovy[cam_id] * np.pi / 180.0
        except:
            fovy_rad = 45 * np.pi / 180.0
    else:
        # From XML: front camera has fovy="45"
        fovy_rad = 45 * np.pi / 180.0
    
    # Convert field of view to focal length
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy  # Assume square pixels
    
    # Principal point (center of image)
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsics = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'width': width, 'height': height,
        'fovy_degrees': fovy_rad * 180.0 / np.pi
    }
    
    return intrinsics

def get_real_camera_intrinsics(calib_file_path="/home/hk/Documents/ACT_Shaka/l515_calibration.npz", 
                              height=480, width=640):
    """Get camera intrinsic parameters for real L515 camera from calibration file"""
    try:
        calib_data = np.load(calib_file_path)
        camera_matrix = calib_data['camera_matrix']  # 3x3 matrix
        
        # Extract intrinsic parameters from camera matrix
        fx = camera_matrix[0, 0]  # 605.10464742
        fy = camera_matrix[1, 1]  # 604.96544362  
        cx = camera_matrix[0, 2]  # 327.29071014
        cy = camera_matrix[1, 2]  # 251.00504208
        
        intrinsics = {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': width, 'height': height,
            'calib_source': 'L515_checkerboard_calibration',
            'rms_error': float(calib_data['rms_error'])
        }
        
        print(f"✓ Loaded L515 calibration: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  RMS calibration error: {intrinsics['rms_error']:.4f} pixels")
        
        return intrinsics
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load L515 calibration ({e})")
        print("   Falling back to estimated intrinsics")
        
        # Fallback: reasonable estimates for L515
        fx = fy = 600.0  # Typical for L515 at 640x480
        cx = width / 2.0
        cy = height / 2.0
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': width, 'height': height,
            'calib_source': 'estimated_fallback'
        }

def depth_to_pointcloud(depth_image, intrinsics, max_depth=2.0, subsample=2):
    """Convert depth image to 3D point cloud using camera intrinsics"""
    height, width = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Create pixel coordinate grids
    u_coords, v_coords = np.meshgrid(
        np.arange(0, width, subsample),
        np.arange(0, height, subsample)
    )
    
    # Subsample depth image to match coordinate grids
    depth_sub = depth_image[::subsample, ::subsample]
    
    # Convert depth from millimeters to meters
    depth_meters = depth_sub.astype(np.float32) / 1000.0
    
    # Filter out invalid depths
    valid_mask = (depth_meters > 0) & (depth_meters < max_depth) & np.isfinite(depth_meters)
    
    # Extract valid pixels and depths
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]  
    z_valid = depth_meters[valid_mask]
    
    # Back-project to 3D using pinhole camera model
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    # Stack into Nx3 point cloud
    points = np.stack([x_valid, y_valid, z_valid], axis=1)
    
    return points

def get_camera_pose(physics, camera_name):
    """Get 4x4 transformation matrix from camera frame to world frame"""
    try:
        from scipy.spatial.transform import Rotation
        
        # Get camera position and orientation from MuJoCo
        cam_id = physics.model.name2id(camera_name, 'camera')
        cam_pos = physics.model.cam_pos[cam_id].copy()
        cam_quat = physics.model.cam_quat[cam_id].copy()  # [w, x, y, z] format
        
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]).as_matrix()
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        
        return T
        
    except Exception as e:
        print(f"Warning: Could not get camera pose for {camera_name}: {e}")
        return np.eye(4)

def points_to_world_frame(points_camera, camera_pose):
    """Transform points from camera frame to world frame"""
    if len(points_camera) == 0:
        return np.empty((0, 3))
    
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
    
    # Transform to world frame
    points_world_homo = (camera_pose @ points_homo.T).T
    
    # Convert back to 3D
    points_world = points_world_homo[:, :3]
    
    return points_world

def process_episode_pointclouds(episode_path, task_name='sim_pick_cube_single_teleop', force=False):
    """Process a single episode to add point cloud data"""
    
    print(f"Processing: {episode_path}")
    
    # Detect episode type from file path or HDF5 attributes
    is_sim_episode = 'sim_episodes' in str(episode_path)
    is_real_episode = 'real_episodes' in str(episode_path)
    
    # Check HDF5 attributes as backup
    with h5py.File(episode_path, 'r') as f:
        if 'sim' in f.attrs:
            is_sim_episode = f.attrs['sim']
            is_real_episode = not is_sim_episode
        elif 'episode_type' in f.attrs:
            episode_type = f.attrs['episode_type'].decode() if isinstance(f.attrs['episode_type'], bytes) else f.attrs['episode_type']
            is_sim_episode = (episode_type == 'sim')
            is_real_episode = (episode_type == 'real')
    
    print(f"  Episode type: {'SIM' if is_sim_episode else 'REAL' if is_real_episode else 'UNKNOWN'}")
    
    # Setup simulation environment for sim episodes (needed for camera pose)
    physics = None
    if is_sim_episode:
        env = make_sim_env(task_name)
        physics = getattr(env, '_physics', getattr(env, 'physics', None))
        
        if physics is None:
            print("⚠️ Warning: No physics interface available for sim episode")
            return False
    
    # Get camera configuration
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    depth_cameras = [cam for cam in camera_names if cam.endswith('_depth')]
    
    if not depth_cameras:
        print("⚠️ No depth cameras found in configuration")
        return False
    
    # Open episode file in read-write mode
    with h5py.File(episode_path, 'r+') as f:
        
        # Check if point clouds already exist
        if 'observations/pointclouds' in f and not force:
            print("✓ Point clouds already exist, skipping (use --force to overwrite)")
            return True
        elif 'observations/pointclouds' in f and force:
            print("🔄 Point clouds exist, but --force specified, removing and recreating...")
            del f['observations/pointclouds']
        
        print(f"Adding point cloud data for cameras: {depth_cameras}")
        
        # Create pointcloud group
        obs = f['observations']
        pointcloud_group = obs.create_group('pointclouds')
        
        # Process each depth camera
        for depth_cam_name in depth_cameras:
            rgb_cam_name = depth_cam_name.replace('_depth', '')
            
            if f'images/{depth_cam_name}' not in obs:
                print(f"⚠️ Warning: No depth data for {depth_cam_name}")
                continue
            
            # Check if this camera's point cloud already exists
            if depth_cam_name in pointcloud_group:
                print(f"✓ Point clouds for {depth_cam_name} already exist, skipping")
                continue
            
            # Load depth images
            depth_data = obs[f'images/{depth_cam_name}'][()]  # Shape: (timesteps, height, width)
            num_timesteps = depth_data.shape[0]
            
            print(f"Processing {num_timesteps} timesteps for {depth_cam_name}")
            
            # Get camera intrinsics based on episode type
            if is_sim_episode:
                # Use MuJoCo camera parameters
                intrinsics = get_camera_intrinsics(rgb_cam_name, 480, 640, physics)
                camera_pose = get_camera_pose(physics, rgb_cam_name)
                print(f"  Using sim camera intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
            else:
                # Use real L515 calibrated parameters
                intrinsics = get_real_camera_intrinsics()
                # For real episodes, we don't have world frame transformation (camera is fixed)
                camera_pose = np.eye(4)  # Identity - points stay in camera frame
                print(f"  Using real L515 calibrated intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
            
            # Process all timesteps
            all_points_camera = []
            all_points_world = []
            all_num_points = []
            
            for t in range(num_timesteps):
                depth_image = depth_data[t]  # Shape: (480, 640)
                
                # Generate point cloud from depth
                points_camera = depth_to_pointcloud(depth_image, intrinsics, max_depth=2.0, subsample=2)
                
                if len(points_camera) > 0:
                    # Transform to world frame (for sim) or keep in camera frame (for real)
                    points_world = points_to_world_frame(points_camera, camera_pose)
                else:
                    points_world = np.empty((0, 3))
                
                all_points_camera.append(points_camera)
                all_points_world.append(points_world)
                all_num_points.append(len(points_camera))
                
                if t % 50 == 0:
                    print(f"  Processed timestep {t}/{num_timesteps}, {len(points_camera)} points")
            
            # Determine maximum points across all timesteps
            max_points = max(all_num_points) if all_num_points else 0
            max_points = max(max_points, 1000)  # Minimum allocation
            
            print(f"  Creating datasets with max_points={max_points}")
            
            # Create datasets for this camera
            pc_group = pointcloud_group.create_group(depth_cam_name)
            
            points_camera_dataset = pc_group.create_dataset(
                'points_camera', (num_timesteps, max_points, 3), 
                dtype=np.float32, chunks=True, fillvalue=0
            )
            points_world_dataset = pc_group.create_dataset(
                'points_world', (num_timesteps, max_points, 3), 
                dtype=np.float32, chunks=True, fillvalue=0
            )
            num_points_dataset = pc_group.create_dataset(
                'num_points', (num_timesteps,), dtype=np.int32, fillvalue=0
            )
            
            # Fill datasets
            for t in range(num_timesteps):
                points_cam = all_points_camera[t]
                points_world = all_points_world[t]
                num_points = all_num_points[t]
                
                if num_points > 0:
                    # Truncate if too many points
                    actual_points = min(num_points, max_points)
                    points_camera_dataset[t, :actual_points, :] = points_cam[:actual_points]
                    points_world_dataset[t, :actual_points, :] = points_world[:actual_points]
                    num_points_dataset[t] = actual_points
                else:
                    num_points_dataset[t] = 0
            
            print(f"✓ Saved point clouds for {depth_cam_name}")
    
    print(f"✓ Episode processing completed: {episode_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate point clouds from recorded depth data')
    parser.add_argument('--dataset_dir', required=True, help='Directory containing episode files')
    parser.add_argument('--episode_type', choices=['sim', 'real', 'both'], default='sim', 
                       help='Which episodes to process')
    parser.add_argument('--episode_nums', nargs='*', type=int, help='Specific episode numbers to process')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if point clouds already exist')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Determine which directories to process
    if args.episode_type == 'sim':
        episode_dirs = [dataset_dir / 'sim_episodes']
    elif args.episode_type == 'real':
        episode_dirs = [dataset_dir / 'real_episodes']
    else:  # both
        episode_dirs = [dataset_dir / 'sim_episodes', dataset_dir / 'real_episodes']
    
    # Process episodes
    for episode_dir in episode_dirs:
        if not episode_dir.exists():
            print(f"Directory not found: {episode_dir}")
            continue
        
        # Find episode files
        if args.episode_nums:
            episode_files = [episode_dir / f'episode_{num}.hdf5' for num in args.episode_nums]
            episode_files = [f for f in episode_files if f.exists()]
        else:
            episode_files = sorted(episode_dir.glob('episode_*.hdf5'))
        
        print(f"\nFound {len(episode_files)} episodes in {episode_dir}")
        
        for episode_file in episode_files:
            try:
                success = process_episode_pointclouds(str(episode_file), force=args.force)
                if not success:
                    print(f"⚠️ Failed to process {episode_file}")
            except Exception as e:
                print(f"❌ Error processing {episode_file}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    main()