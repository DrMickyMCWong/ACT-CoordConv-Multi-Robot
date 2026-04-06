"""
Point Cloud Visualization Helper Functions for Jupyter Notebooks

Add these functions to your replay notebook to visualize 3D point clouds from depth cameras.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_pointcloud_from_hdf5(dataset_path, camera_name='front_depth', timestep=0):
    """
    Load point cloud data from HDF5 file.
    
    Args:
        dataset_path: Path to HDF5 episode file
        camera_name: Name of depth camera (e.g., 'front_depth')  
        timestep: Which timestep to load
    
    Returns:
        points_camera: Nx3 points in camera frame
        points_world: Nx3 points in world frame
        num_points: Number of valid points
    """
    import h5py
    
    with h5py.File(dataset_path, 'r') as f:
        if f'observations/pointclouds/{camera_name}' not in f:
            raise ValueError(f"No point cloud data for camera '{camera_name}'")
        
        pc_group = f[f'observations/pointclouds/{camera_name}']
        
        # Load data
        points_camera_full = pc_group['points_camera'][timestep]
        points_world_full = pc_group['points_world'][timestep]  
        num_points = pc_group['num_points'][timestep]
        
        # Extract only valid points (first num_points)
        points_camera = points_camera_full[:num_points]
        points_world = points_world_full[:num_points]
        
        print(f"Loaded {num_points} points from {camera_name} at timestep {timestep}")
        
        return points_camera, points_world, num_points

def visualize_pointcloud_3d(points, colors=None, title="3D Point Cloud", max_points=5000):
    """
    Visualize 3D point cloud using matplotlib with proper camera orientation.
    
    Coordinate system:
    - X: Along monitor width (left-right) 
    - Y: Into the screen (forward-back, depth direction)
    - Z: Along monitor height (up-down)
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors or Nx1 array for colormap
        title: Plot title
        max_points: Maximum points to plot (for performance)
    """
    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by depth if no colors provided
    if colors is None:
        colors = points[:, 2]  # Use Z coordinate for coloring
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=1, cmap='viridis')
    
    # Set proper labels for intuitive viewing
    ax.set_xlabel('X (meters) ← → (width)')
    ax.set_ylabel('Y (meters) ↗ ↙ (depth)')  
    ax.set_zlabel('Z (meters) ↕ (height)')
    ax.set_title(f'{title} ({len(points)} points)')
    
    # Set viewing angle for better visualization
    # elev: elevation angle (looking down from above)
    # azim: azimuth angle (rotation around Z-axis)
    ax.view_init(elev=20, azim=45)  # Slightly above, rotated for good perspective
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Height Z (m)')
    plt.tight_layout()
    plt.show()

def visualize_pointcloud_2d_views(points, colors=None, title="Point Cloud Views"):
    """
    Show 2D projections of the 3D point cloud with X along monitor width.
    
    View orientations:
    - Top-left: XY view (top-down, X=horizontal, Y=vertical)  
    - Top-right: XZ view (front view, X=horizontal, Z=vertical)
    - Bottom-left: YZ view (side view, Y=horizontal, Z=vertical)
    - Bottom-right: Depth histogram
    """
    if colors is None:
        colors = points[:, 2]  # Use depth for coloring
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top view (XY) - X horizontal, Y vertical - looking down from above
    im1 = axes[0,0].scatter(points[:, 0], points[:, 1], c=colors, s=1, cmap='viridis')
    axes[0,0].set_xlabel('X (meters) →')  # Horizontal axis
    axes[0,0].set_ylabel('Y (meters) ↑')  # Vertical axis
    axes[0,0].set_title('Top View (XY) - Looking Down')
    axes[0,0].axis('equal')
    axes[0,0].grid(True)
    
    # Front view (XZ) - X horizontal, Z vertical - looking from front
    im2 = axes[0,1].scatter(points[:, 0], points[:, 2], c=colors, s=1, cmap='viridis')
    axes[0,1].set_xlabel('X (meters) →')  # Horizontal axis
    axes[0,1].set_ylabel('Z (meters) ↑')  # Vertical axis (height)
    axes[0,1].set_title('Front View (XZ) - Looking Forward')
    axes[0,1].axis('equal')
    axes[0,1].grid(True)
    
    # Side view (YZ) - Y horizontal, Z vertical - looking from side
    im3 = axes[1,0].scatter(points[:, 1], points[:, 2], c=colors, s=1, cmap='viridis')
    axes[1,0].set_xlabel('Y (meters) →')  # Horizontal axis (depth/forward)
    axes[1,0].set_ylabel('Z (meters) ↑')  # Vertical axis (height)
    axes[1,0].set_title('Side View (YZ) - Looking from Side')
    axes[1,0].axis('equal')
    axes[1,0].grid(True)
    
    # Histogram of depths
    axes[1,1].hist(points[:, 2], bins=50, alpha=0.7, color='blue')
    axes[1,1].set_xlabel('Depth Z (meters)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Depth Distribution')
    axes[1,1].grid(True)
    
    plt.suptitle(f'{title} ({len(points)} points)')
    plt.tight_layout()
    plt.show()

def compare_timesteps_pointclouds(dataset_path, camera_name='front_depth', timesteps=[0, 10, 20]):
    """
    Compare point clouds from different timesteps to see movement.
    X is horizontal across monitor, Y is depth, Z is height.
    """
    fig = plt.figure(figsize=(15, 5))
    
    for i, timestep in enumerate(timesteps):
        try:
            points_camera, points_world, num_points = load_pointcloud_from_hdf5(
                dataset_path, camera_name, timestep
            )
            
            ax = fig.add_subplot(1, len(timesteps), i+1, projection='3d')
            colors = points_camera[:, 2]  # Color by height (Z)
            
            scatter = ax.scatter(points_camera[:, 0], points_camera[:, 1], points_camera[:, 2], 
                               c=colors, s=1, cmap='viridis')
            
            # Set proper labels and viewing angle
            ax.set_xlabel('X ← →')  # Width (horizontal)
            ax.set_ylabel('Y ↗ ↙')  # Depth (into screen)
            ax.set_zlabel('Z ↕')    # Height (vertical)
            ax.set_title(f'Step {timestep}\n({num_points} pts)')
            
            # Set consistent axis limits for comparison
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([0.2, 1.2])
            
            # Set good viewing angle for all subplots
            ax.view_init(elev=15, azim=45)
            
        except Exception as e:
            print(f"Error loading timestep {timestep}: {e}")
    
    plt.suptitle(f'Point Cloud Evolution - {camera_name}')
    plt.tight_layout()
    plt.show()

def analyze_manipulation_area(points_camera, workspace_bounds=None):
    """
    Analyze point cloud within the manipulation workspace (A4 checkerboard area).
    
    Args:
        points_camera: Nx3 points in camera frame
        workspace_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
    
    Returns:
        workspace_points: Points within the manipulation area
        statistics: Dictionary with analysis results
    """
    if workspace_bounds is None:
        # Default A4 workspace bounds (adjust based on your setup)
        workspace_bounds = {
            'x_min': -0.15, 'x_max': 0.15,   # ±15cm in X
            'y_min': -0.10, 'y_max': 0.10,   # ±10cm in Y  
            'z_min': 0.3,   'z_max': 1.0     # 30cm to 1m depth
        }
    
    # Filter points within workspace
    mask = (
        (points_camera[:, 0] >= workspace_bounds['x_min']) & 
        (points_camera[:, 0] <= workspace_bounds['x_max']) &
        (points_camera[:, 1] >= workspace_bounds['y_min']) & 
        (points_camera[:, 1] <= workspace_bounds['y_max']) &
        (points_camera[:, 2] >= workspace_bounds['z_min']) & 
        (points_camera[:, 2] <= workspace_bounds['z_max'])
    )
    
    workspace_points = points_camera[mask]
    
    stats = {
        'total_points': len(points_camera),
        'workspace_points': len(workspace_points),
        'workspace_percentage': len(workspace_points) / len(points_camera) * 100,
        'workspace_bounds': workspace_bounds
    }
    
    if len(workspace_points) > 0:
        stats.update({
            'mean_position': np.mean(workspace_points, axis=0),
            'std_position': np.std(workspace_points, axis=0),
            'depth_range': [workspace_points[:, 2].min(), workspace_points[:, 2].max()],
            'depth_mean': workspace_points[:, 2].mean()
        })
    
    print(f"Workspace Analysis:")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Workspace points: {stats['workspace_points']} ({stats['workspace_percentage']:.1f}%)")
    if len(workspace_points) > 0:
        print(f"  Mean position: [{stats['mean_position'][0]:.3f}, {stats['mean_position'][1]:.3f}, {stats['mean_position'][2]:.3f}]")
        print(f"  Depth range: {stats['depth_range'][0]:.3f} - {stats['depth_range'][1]:.3f}m")
    
    return workspace_points, stats

# Example usage for notebook:
"""
# 1. Load point cloud data
dataset_path = '/path/to/episode_0.hdf5'
points_camera, points_world, num_points = load_pointcloud_from_hdf5(dataset_path)

# 2. Visualize in 3D
visualize_pointcloud_3d(points_camera, title="Camera Frame Point Cloud")

# 3. Show 2D projections  
visualize_pointcloud_2d_views(points_camera, title="Point Cloud Projections")

# 4. Compare multiple timesteps
compare_timesteps_pointclouds(dataset_path, timesteps=[0, 5, 10, 15])

# 5. Analyze manipulation area
workspace_points, stats = analyze_manipulation_area(points_camera)
visualize_pointcloud_3d(workspace_points, title="Manipulation Workspace Points")
"""