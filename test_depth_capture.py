#!/usr/bin/env python3
"""
Quick test to verify L515 depth capture is working
Run this before the full recording to debug depth issues
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_capture():
    """Test L515 depth capture with the exact same logic as the recording script"""
    
    print("Testing L515 depth capture...")
    
    # Create context and find L515
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("ERROR: No RealSense devices found!")
        return False
    
    device = devices[0]
    print(f"Device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable device
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    
    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        print("✓ Pipeline started")
        
        # Get depth sensor info
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale:.6f}")
        
        if depth_sensor.supports(rs.option.depth_offset):
            depth_offset = depth_sensor.get_option(rs.option.depth_offset)
            print(f"Depth offset: {depth_offset*1000:.1f}mm")
        
        # Warm up
        for i in range(30):
            pipeline.wait_for_frames()
        
        print("\nTesting depth capture (press 'q' to quit, 's' to save sample)...")
        
        align_to_color = rs.align(rs.stream.color)
        
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not aligned_depth_frame:
                print("Warning: Missing frames")
                continue
            
            # Process RGB
            rgb_array = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
            
            # Test depth processing with both methods
            # Method 1: RealSense get_distance (accurate but slow for full frame)
            center_x, center_y = 320, 240
            center_dist_mm = aligned_depth_frame.get_distance(center_x, center_y) * 1000
            
            # Method 2: Raw data conversion
            depth_raw = np.asanyarray(aligned_depth_frame.get_data())
            depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000
            center_calc_mm = depth_mm[center_y, center_x]
            
            # Statistics
            valid_pixels = np.count_nonzero(depth_mm)
            if valid_pixels > 0:
                valid_depths = depth_mm[depth_mm > 0]
                min_depth, max_depth, mean_depth = valid_depths.min(), valid_depths.max(), valid_depths.mean()
            else:
                min_depth = max_depth = mean_depth = 0
            
            # Create visualization
            depth_colorized = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.1), cv2.COLORMAP_JET
            )
            
            # Add text overlay
            cv2.putText(rgb_image, f"Center: API={center_dist_mm:.1f}mm, Calc={center_calc_mm:.1f}mm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Valid pixels: {valid_pixels} / {depth_mm.size}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Range: {min_depth:.0f}-{max_depth:.0f}mm (avg: {mean_depth:.0f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mark center point
            cv2.circle(rgb_image, (center_x, center_y), 5, (255, 0, 0), 2)
            cv2.circle(depth_colorized, (center_x, center_y), 5, (255, 255, 255), 2)
            
            # Display side by side
            combined = np.hstack((rgb_image, cv2.cvtColor(depth_colorized, cv2.COLOR_BGR2RGB)))
            cv2.imshow('L515 Test: RGB | Depth', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save test data
                print(f"\nSaving test data:")
                print(f"  Valid pixels: {valid_pixels} / {depth_mm.size}")
                print(f"  Depth range: {min_depth:.1f} - {max_depth:.1f}mm")
                print(f"  Center depth: API={center_dist_mm:.1f}mm, Calc={center_calc_mm:.1f}mm")
                
                np.save('test_depth_mm.npy', depth_mm.astype(np.uint16))
                cv2.imwrite('test_rgb.png', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print("  Saved: test_depth_mm.npy, test_rgb.png")
                
                return valid_pixels > 0  # Return True if we got valid depth data
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
    return False

if __name__ == '__main__':
    success = test_depth_capture()
    if success:
        print("\n✓ Depth capture test PASSED - ready for recording")
    else:
        print("\n✗ Depth capture test FAILED - check L515 connection and setup")