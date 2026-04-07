#!/usr/bin/env python3
"""
Visualize saved L515 capture data
- Load and display RGB, depth, and camera parameters
- Create comparison plots and 3D visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_l515_data():
    """Load and visualize saved L515 data"""
    
    print("=" * 70)
    print("Loading L515 Saved Data")
    print("=" * 70)
    
    # Load data
    rgb = cv2.imread('l515_rgb.png')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    depth_colormap = cv2.imread('l515_depth_colormap.png')
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
    
    depth_aligned = np.load('l515_depth_aligned.npy')
    camera_params = np.load('l515_camera_params.npy', allow_pickle=True).item()
    
    print(f"\n✓ Loaded:")
    print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  Depth shape: {depth_aligned.shape}, dtype: {depth_aligned.dtype}")
    print(f"  Depth range: {depth_aligned.min()}mm - {depth_aligned.max()}mm")
    print(f"  Depth mean (valid pixels): {depth_aligned[depth_aligned > 0].mean():.1f}mm")
    
    # Print camera parameters
    print("\n" + "=" * 70)
    print("CAMERA PARAMETERS")
    print("=" * 70)
    intrinsics = camera_params['color_intrinsics']
    print(f"Resolution: {intrinsics['width']} x {intrinsics['height']}")
    print(f"Focal Length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
    print(f"Principal Point: cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. RGB Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title('RGB Image (640x480)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Depth Colormap
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(depth_colormap)
    ax2.set_title('Depth Colormap (Saved)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Depth Raw - Better colormap
    ax3 = plt.subplot(2, 3, 3)
    depth_display = depth_aligned.copy().astype(float)
    depth_display[depth_display == 0] = np.nan  # Set invalid to NaN for better viz
    im3 = ax3.imshow(depth_display, cmap='jet', vmin=200, vmax=2000)
    ax3.set_title(f'Depth Raw (200-2000mm)\nMin: {depth_aligned[depth_aligned>0].min()}mm, Max: {depth_aligned.max()}mm', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='Depth (mm)', fraction=0.046)
    
    # 4. Depth Histogram
    ax4 = plt.subplot(2, 3, 4)
    valid_depth = depth_aligned[depth_aligned > 0]
    ax4.hist(valid_depth, bins=100, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(valid_depth.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_depth.mean():.1f}mm')
    ax4.axvline(np.median(valid_depth), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_depth):.1f}mm')
    ax4.set_xlabel('Depth (mm)', fontsize=12)
    ax4.set_ylabel('Pixel Count', fontsize=12)
    ax4.set_title('Depth Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. RGB with Depth Overlay
    ax5 = plt.subplot(2, 3, 5)
    # Create overlay: blend RGB with depth
    rgb_normalized = rgb.astype(float) / 255.0
    depth_for_overlay = depth_aligned.copy().astype(float)
    depth_for_overlay[depth_for_overlay == 0] = np.nan
    depth_normalized = (depth_for_overlay - 200) / (2000 - 200)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Apply colormap to depth
    depth_colored = plt.cm.jet(depth_normalized)[:, :, :3]
    
    # Blend
    alpha = 0.5
    blended = alpha * rgb_normalized + (1 - alpha) * depth_colored
    ax5.imshow(blended)
    ax5.set_title('RGB + Depth Overlay (50/50)', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # 6. Depth Cross-sections
    ax6 = plt.subplot(2, 3, 6)
    h, w = depth_aligned.shape
    # Horizontal cross-section (middle row)
    h_cross = depth_aligned[h//2, :]
    # Vertical cross-section (middle column)
    v_cross = depth_aligned[:, w//2]
    
    ax6.plot(h_cross, label=f'Horizontal (row {h//2})', linewidth=2, alpha=0.7)
    ax6.plot(v_cross, label=f'Vertical (col {w//2})', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Pixel', fontsize=12)
    ax6.set_ylabel('Depth (mm)', fontsize=12)
    ax6.set_title('Depth Cross-Sections (Center)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, depth_aligned.max() + 100])
    
    plt.tight_layout()
    plt.savefig('l515_visualization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: l515_visualization_comparison.png")
    plt.show()
    
    # Create 3D point cloud visualization (downsampled)
    print("\n" + "=" * 70)
    print("Generating 3D Point Cloud Visualization...")
    print("=" * 70)
    
    fig2 = plt.figure(figsize=(16, 8))
    
    # Get camera intrinsics
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Generate point cloud (downsampled for speed)
    step = 8  # Use every 8th pixel
    points_3d = []
    colors = []
    
    for v in range(0, depth_aligned.shape[0], step):
        for u in range(0, depth_aligned.shape[1], step):
            z = depth_aligned[v, u]
            if z > 0 and z < 3000:  # Valid depth
                # Backproject to 3D
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d.append([x, y, z])
                colors.append(rgb[v, u] / 255.0)
    
    points_3d = np.array(points_3d)
    colors = np.array(colors)
    
    print(f"Point cloud: {len(points_3d)} points")
    
    # Two views of the point cloud
    for idx, (elev, azim, title) in enumerate([
        (20, 45, '3D Point Cloud - View 1 (Top-Front)'),
        (10, 135, '3D Point Cloud - View 2 (Side)')
    ]):
        ax = fig2.add_subplot(1, 2, idx+1, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c=colors, s=1, alpha=0.5)
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_zlabel('Z (mm)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio
        max_range = np.array([
            points_3d[:, 0].max() - points_3d[:, 0].min(),
            points_3d[:, 1].max() - points_3d[:, 1].min(),
            points_3d[:, 2].max() - points_3d[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('l515_point_cloud_3d.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: l515_point_cloud_3d.png")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("DEPTH STATISTICS")
    print("=" * 70)
    valid_pixels = np.sum(depth_aligned > 0)
    total_pixels = depth_aligned.size
    print(f"Valid pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    print(f"Min depth: {depth_aligned[depth_aligned>0].min():.1f}mm ({depth_aligned[depth_aligned>0].min()/10:.1f}cm)")
    print(f"Max depth: {depth_aligned.max():.1f}mm ({depth_aligned.max()/10:.1f}cm)")
    print(f"Mean depth: {valid_depth.mean():.1f}mm ({valid_depth.mean()/10:.1f}cm)")
    print(f"Median depth: {np.median(valid_depth):.1f}mm ({np.median(valid_depth)/10:.1f}cm)")
    print(f"Std dev: {valid_depth.std():.1f}mm ({valid_depth.std()/10:.1f}cm)")
    
    # Depth ranges
    print("\nDepth Distribution:")
    ranges = [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500), (1500, 2000), (2000, 5000)]
    for r_min, r_max in ranges:
        count = np.sum((depth_aligned >= r_min) & (depth_aligned < r_max))
        pct = 100 * count / valid_pixels if valid_pixels > 0 else 0
        print(f"  {r_min:4d}-{r_max:4d}mm: {count:6d} pixels ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    visualize_l515_data()
