#!/usr/bin/env python3
"""
Comprehensive camera diagnostic to identify why policy fails.
Compares training data vs live camera vs DemoGen recording method.
"""

import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_hdf5_images(dataset_path):
    """Load images from HDF5 file."""
    if not os.path.isfile(dataset_path):
        print(f'❌ File not found: {dataset_path}')
        return None
    
    with h5py.File(dataset_path, 'r') as root:
        images = root['/observations/images/front'][()]
    return images

def capture_rgb_method1(num_frames=10):
    """
    Method 1: record_episodes_custom.py / evaluate_custom.py
    Uses cv2.VideoCapture with BGR→RGB conversion
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return None
    
    frames = []
    print("📷 Capturing with Method 1 (VideoCapture BGR→RGB)...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            # EXACT processing from record_episodes_custom.py
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            frame = frame[90:, 0:510]  # Crop
            frame = cv2.resize(frame, (640, 480))  # Resize
            frames.append(frame)
    
    cap.release()
    return np.array(frames) if frames else None

def capture_rgbd_demogen_method(num_frames=10):
    """
    Method 2: record_episodes_custom_depth_cam.py (DemoGen)
    Uses pyrealsense2 with proper color handling
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("⚠️  pyrealsense2 not available, skipping DemoGen method")
        return None
    
    print("📷 Capturing with Method 2 (RealSense RGB from RGBD)...")
    
    # Initialize pipeline (same as record_episodes_custom_depth_cam.py)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # Short Range
    
    depth_scale = depth_sensor.get_depth_scale()
    depth_offset_mm = 0
    if depth_sensor.supports(rs.option.depth_offset):
        depth_offset_mm = depth_sensor.get_option(rs.option.depth_offset) * 1000
    
    align_to_color = rs.align(rs.stream.color)
    
    # Warm up
    for _ in range(30):
        frames = pipeline.wait_for_frames()
    
    # Capture frames
    rgb_frames = []
    for i in range(num_frames):
        frames = pipeline.wait_for_frames()
        aligned_frames = align_to_color.process(frames)
        color_frame = aligned_frames.get_color_frame()
        
        if color_frame:
            # Convert to numpy (BGR from camera)
            color_image_bgr = np.asanyarray(color_frame.get_data())
            # Convert BGR to RGB (same as DemoGen script)
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply same crop/resize
            color_cropped = color_image_rgb[0:480, 90:600]  # Match DemoGen crop
            color_resized = cv2.resize(color_cropped, (640, 480), interpolation=cv2.INTER_AREA)
            rgb_frames.append(color_resized)
    
    pipeline.stop()
    return np.array(rgb_frames) if rgb_frames else None

def analyze_video(name, video):
    """Analyze video statistics."""
    print(f"\n{'='*60}")
    print(f"📹 {name}")
    print(f"{'='*60}")
    print(f"  Shape: {video.shape}")
    print(f"  Channels: {video.shape[-1]}")
    print(f"  Dtype: {video.dtype}")
    print(f"  Range: [{video.min()}, {video.max()}]")
    print(f"  Mean: {video.mean():.2f}")
    print(f"  Std: {video.std():.2f}")
    
    if video.shape[-1] >= 3:
        r_mean = video[:,:,:,0].mean()
        g_mean = video[:,:,:,1].mean()
        b_mean = video[:,:,:,2].mean()
        print(f"  Channel means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}")
        
        # Check if BGR instead of RGB (common mistake indicator)
        if abs(r_mean - b_mean) > 20:
            print(f"  ⚠️  Large R-B difference ({abs(r_mean - b_mean):.1f}) - possible BGR/RGB mismatch")
    
    if video.shape[-1] == 4:
        print(f"  ⚠️  4 channels (RGBD) - Depth mean: {video[:,:,:,3].mean():.2f}")
    
    # Sample pixel values
    h, w = video.shape[1], video.shape[2]
    print(f"  Sample pixels:")
    print(f"    Top-left [0,0]: {video[0, 0, 0, :]}")
    print(f"    Center [{h//2},{w//2}]: {video[0, h//2, w//2, :]}")
    print(f"    Top-right [0,{w-1}]: {video[0, 0, w-1, :]}")

def compare_first_frames(videos_dict):
    """Visual comparison of first frames."""
    num_videos = len(videos_dict)
    fig, axes = plt.subplots(1, num_videos, figsize=(5*num_videos, 5))
    
    if num_videos == 1:
        axes = [axes]
    
    for idx, (name, video) in enumerate(videos_dict.items()):
        first_frame = video[0]
        
        # Handle RGBD
        if video.shape[-1] == 4:
            display_frame = first_frame[:, :, :3]  # RGB only
            title = f"{name}\n(RGBD - RGB shown)"
        else:
            display_frame = first_frame
            title = f"{name}\n({video.shape[-1]} channels)"
        
        axes[idx].imshow(display_frame)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')
        
        # Add mean value as text
        mean_val = display_frame.mean()
        axes[idx].text(10, 30, f"Mean: {mean_val:.1f}", 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=9)
    
    plt.tight_layout()
    save_path = '/home/hk/Documents/ACT_Shaka/camera_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison saved to: {save_path}")
    plt.close()

def main():
    print("="*80)
    print("CAMERA STREAM DIAGNOSTIC - Identifying Policy Failure Root Cause")
    print("="*80)
    print("\nThis will compare:")
    print("1. Training data (task8/episode_1.hdf5)")
    print("2. Evaluation data (latest eval)")
    print("3. Live camera - Method 1 (record_episodes_custom.py method)")
    print("4. Live camera - Method 2 (DemoGen RGBD method, RGB only)")
    print("="*80)
    
    videos = {}
    
    # 1. Training data
    print("\n[1/4] Loading training data...")
    training_path = '/home/hk/Documents/ACT_Shaka/data/task8/episode_1.hdf5'
    training_video = load_hdf5_images(training_path)
    if training_video is not None:
        videos['Training (episode_1)'] = training_video
        analyze_video('Training Data', training_video)
    
    # 2. Evaluation data
    print("\n[2/4] Loading evaluation data...")
    eval_path = '/home/hk/Documents/ACT_Shaka/data/evaluation/task8/eval_result_task8_20260117_151219.hdf5'
    if os.path.exists(eval_path):
        eval_video = load_hdf5_images(eval_path)
        if eval_video is not None:
            videos['Evaluation'] = eval_video
            analyze_video('Evaluation Data', eval_video)
    
    # 3. Live camera - Method 1 (what record_episodes_custom.py uses)
    print("\n[3/4] Testing live camera - Method 1 (VideoCapture)...")
    method1_video = capture_rgb_method1(num_frames=10)
    if method1_video is not None:
        videos['Live - Method 1 (VideoCapture)'] = method1_video
        analyze_video('Live Camera - Method 1', method1_video)
    
    # 4. Live camera - Method 2 (DemoGen method)
    print("\n[4/4] Testing live camera - Method 2 (RealSense RGBD→RGB)...")
    method2_video = capture_rgbd_demogen_method(num_frames=10)
    if method2_video is not None:
        videos['Live - Method 2 (RealSense)'] = method2_video
        analyze_video('Live Camera - Method 2 (DemoGen)', method2_video)
    
    # Generate comparison
    if len(videos) >= 2:
        print("\n" + "="*80)
        print("GENERATING VISUAL COMPARISON...")
        print("="*80)
        compare_first_frames(videos)
    
    # Analysis
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    if 'Training (episode_1)' in videos and 'Live - Method 1 (VideoCapture)' in videos:
        train_mean = videos['Training (episode_1)'].mean()
        live1_mean = videos['Live - Method 1 (VideoCapture)'].mean()
        
        print(f"\n📊 Key Metrics:")
        print(f"  Training data mean: {train_mean:.2f}")
        print(f"  Live Method 1 mean: {live1_mean:.2f}")
        print(f"  Difference: {abs(train_mean - live1_mean):.2f}")
        
        if abs(train_mean - live1_mean) > 50:
            print(f"\n❌ MAJOR MISMATCH DETECTED!")
            print(f"   Training data looks very different from live camera")
            print(f"   This explains why policy fails!")
            print(f"\n🔍 Possible causes:")
            print(f"   1. Training data was recorded with DIFFERENT script")
            print(f"   2. Camera settings changed between recording and evaluation")
            print(f"   3. BGR/RGB conversion bug during training recording")
            print(f"   4. Lighting conditions drastically different")
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Re-record training data using record_episodes_custom.py")
            print(f"   Or check if you used record_episodes_custom_depth_cam.py")
        else:
            print(f"\n✅ Training and live camera are similar")
            print(f"   Policy failure is likely NOT due to camera format")
    
    # Check which recording method was used
    print(f"\n🔍 Identifying recording method used for training:")
    if 'Training (episode_1)' in videos:
        train_shape = videos['Training (episode_1)'].shape
        if train_shape[-1] == 4:
            print(f"   ✓ Training has 4 channels → Used record_episodes_custom_depth_cam.py")
            print(f"   ⚠️  But policy expects 3 channels!")
            print(f"   💡 Must retrain with 3-channel data or modify backbone for 4 channels")
        elif train_shape[-1] == 3:
            print(f"   ✓ Training has 3 channels → Used record_episodes_custom.py")
            
            # Compare with both methods
            if 'Live - Method 1 (VideoCapture)' in videos:
                match1 = abs(videos['Training (episode_1)'].mean() - 
                           videos['Live - Method 1 (VideoCapture)'].mean()) < 30
                print(f"   {'✓' if match1 else '✗'} Matches Method 1 (VideoCapture): {match1}")
            
            if 'Live - Method 2 (RealSense)' in videos:
                match2 = abs(videos['Training (episode_1)'].mean() - 
                           videos['Live - Method 2 (RealSense)'].mean()) < 30
                print(f"   {'✓' if match2 else '✗'} Matches Method 2 (RealSense): {match2}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE - Check camera_comparison.png for visual verification")
    print("="*80)

if __name__ == "__main__":
    main()
