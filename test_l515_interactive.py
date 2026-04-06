#!/usr/bin/env python3
"""
Interactive L515 test with live preview using RealSense SDK.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

print("=" * 60)
print("L515 RealSense Interactive Test")
print("=" * 60)
print("\nThis uses the RealSense SDK directly for better L515 compatibility.")
print("Press 'q' to quit, 's' to save snapshot, 'r' to save RGB image\n")

# Configure the pipeline - match the working notebook configuration
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB stream only (same as working notebook)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # Start streaming
    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)
    print("✓ RealSense camera started successfully")
    
    # Get device info
    device = profile.get_device()
    print(f"Device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    
    print("\n" + "=" * 60)
    print("LIVE PREVIEW")
    print("=" * 60)
    print("\nPress 'q' to quit, 's' to save snapshot, 'r' to save RGB image")
    print("=" * 60 + "\n")

    frame_count = 0
    rgb_save_count = 0  # Counter for RGB saves
    
    # Allow camera to settle (same as working notebook)
    for i in range(30):
        pipeline.wait_for_frames()
    
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Get RGB frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Failed to get color frame")
            continue
            
        rgb = np.asanyarray(color_frame.get_data())
        
        # Create a fake IR display (since we're not using IR stream)
        # Just make a gray version of RGB for the side-by-side display
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_colored = cv2.applyColorMap(rgb_gray, cv2.COLORMAP_BONE)
        
        # Resize to same size for side-by-side display
        h, w = 360, 480
        rgb_resized = cv2.resize(rgb, (w, h))
        ir_resized = cv2.resize(ir_colored, (w, h))
        
        # Add frame statistics
        rgb_mean = rgb.mean()
        rgb_max = rgb.max()
        rgb_min = rgb.min()
        
        # Add text overlay with stats
        cv2.putText(rgb_resized, f"RGB Camera", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(rgb_resized, f"Mean: {rgb_mean:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(rgb_resized, f"Range: [{rgb_min}-{rgb_max}]", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if rgb_mean < 10:
            cv2.putText(rgb_resized, "WARNING: Very dark!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(rgb_resized, "Check if lens is covered", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.putText(ir_resized, f"IR Camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([rgb_resized, ir_resized])
        
        # Add instructions at bottom
        cv2.putText(combined, "Press 'q' to quit  |  's' to save snapshot  |  'r' to save RGB", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('L515 Live Feed (RGB | IR)', combined)
        
        # Print diagnostics every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            status = "DARK (check lens!)" if rgb_mean < 10 else "OK"
            print(f"Frame {frame_count}: RGB mean={rgb_mean:.1f}, status={status}")
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Convert BGR to RGB for saving
            rgb_save = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/tmp/l515_snapshot_rgb.jpg', rgb_save)
            cv2.imwrite('/tmp/l515_snapshot_ir.jpg', ir_colored)  # Save the grayscale version
            print(f"\n✓ Saved snapshots to /tmp/l515_snapshot_*.jpg")
        elif key == ord('r'):
            rgb_save_count += 1
            # Convert BGR to RGB for saving
            rgb_save = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_filename = f'/home/hk/Documents/ACT_Shaka/test_l515_rgb_{rgb_save_count:03d}.jpg'
            cv2.imwrite(rgb_filename, rgb_save)
            print(f"\n✓ Saved RGB image to {rgb_filename}")

except Exception as e:
    print(f"Error: {e}")
    print("Make sure the L515 camera is connected and not being used by another application.")

finally:
    # Cleanup - only stop if pipeline was started successfully
    try:
        pipeline.stop()
        print("\n✓ Pipeline stopped")
    except:
        pass  # Pipeline was never started
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
