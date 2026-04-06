#!/usr/bin/env python3
"""
Advanced test script for Intel RealSense L515
- Configures min distance (0.25m)
- Aligns depth to color (resolves resolution mismatch)
- Shows camera intrinsics and extrinsics
- Tests depth accuracy at close range
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_l515_advanced():
    """Test L515 with proper configuration for close-range work"""
    
    # Create context and find L515
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("ERROR: No RealSense devices found!")
        return
    
    device = devices[0]
    print("=" * 70)
    print(f"Device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    print("=" * 70)
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable device
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    
    # Configure streams
    # RGB: 640x480 @ 30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Depth: Use native resolution (L515 typically uses 640x480 or 1024x768)
    # Don't specify resolution - let it use native, then align to color
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    
    print("\nStarting pipeline...")
    pipeline_started = False
    
    try:
        # Start streaming
        profile = pipeline.start(config)
        pipeline_started = True
        
        # Get depth sensor and configure visual preset for close range
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # Get available visual presets
        print("\n" + "=" * 70)
        print("DEPTH SENSOR CONFIGURATION")
        print("=" * 70)
        
        # Check if visual preset option exists
        if depth_sensor.supports(rs.option.visual_preset):
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            current_preset = depth_sensor.get_option(rs.option.visual_preset)
            print(f"Current Visual Preset: {current_preset}")
            print(f"Available presets: {preset_range.min} to {preset_range.max}")
            
            # L515 presets (may vary by firmware):
            # 0 = Custom
            # 1 = Default
            # 2 = Hand
            # 3 = Short Range
            # 4 = Automatic (Medium Range)
            # 5 = Max Range
            
            # Set to Short Range (preset 3) for minimum distance ~0.25m
            try:
                depth_sensor.set_option(rs.option.visual_preset, 3)  # Short Range
                print(f"✓ Set Visual Preset to: 3 (Short Range - min distance ~0.25m)")
            except Exception as e:
                print(f"Warning: Could not set visual preset: {e}")
        
        # CRITICAL: Check and disable depth offset (from your screenshot: was 4.5m!)
        if depth_sensor.supports(rs.option.depth_offset):
            current_offset = depth_sensor.get_option(rs.option.depth_offset)
            print(f"⚠️  Current Depth Offset: {current_offset:.6f} meters = {current_offset*1000:.1f}mm")
            if abs(current_offset) > 0.001:  # If offset is non-zero
                print(f"   This adds {current_offset:.3f}m to all measurements!")
                try:
                    depth_sensor.set_option(rs.option.depth_offset, 0.0)
                    print(f"✓ Reset Depth Offset to: 0.0")
                except Exception as e:
                    print(f"⚠️  Could not reset depth offset: {e}")
            else:
                print(f"✓ Depth Offset is already 0 (good)")
        
        # Check min/max distance settings
        if depth_sensor.supports(rs.option.min_distance):
            min_dist = depth_sensor.get_option(rs.option.min_distance)
            print(f"Min Distance: {min_dist}mm ({min_dist/10:.1f}cm)")
        
        if depth_sensor.supports(rs.option.max_distance):
            max_dist = depth_sensor.get_option(rs.option.max_distance)
            print(f"Max Distance: {max_dist}mm ({max_dist/10:.1f}cm)")
        
        # Get and print depth scale
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {depth_scale:.6f} (units to meters)")
        print(f"  -> Raw sensor value × {depth_scale*1000:.6f} = mm")
        
        # Create align object to align depth to color
        align_to_color = rs.align(rs.stream.color)
        
        # Get intrinsics
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        
        color_intrinsics = color_profile.get_intrinsics()
        depth_intrinsics = depth_profile.get_intrinsics()
        
        print("\n" + "=" * 70)
        print("COLOR CAMERA INTRINSICS")
        print("=" * 70)
        print(f"Resolution: {color_intrinsics.width} x {color_intrinsics.height}")
        print(f"Focal Length: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"Principal Point: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
        print(f"Distortion Model: {color_intrinsics.model}")
        print(f"Distortion Coeffs: {color_intrinsics.coeffs}")
        
        print("\n" + "=" * 70)
        print("DEPTH CAMERA INTRINSICS (Native)")
        print("=" * 70)
        print(f"Resolution: {depth_intrinsics.width} x {depth_intrinsics.height}")
        print(f"Focal Length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
        print(f"Principal Point: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")
        
        # Get extrinsics (depth to color transform)
        try:
            extrinsics = depth_profile.get_extrinsics_to(color_profile)
            print("\n" + "=" * 70)
            print("EXTRINSICS: Depth to Color Transform")
            print("=" * 70)
            print("Rotation matrix:")
            rotation = np.array(extrinsics.rotation).reshape(3, 3)
            print(rotation)
            print(f"\nTranslation (mm): {extrinsics.translation}")
            print(f"Translation (m): [{extrinsics.translation[0]/1000:.4f}, "
                  f"{extrinsics.translation[1]/1000:.4f}, {extrinsics.translation[2]/1000:.4f}]")
        except Exception as e:
            print(f"\nWarning: Could not get extrinsics: {e}")
        
        # Store depth offset for compensation
        depth_offset_mm = 0
        if depth_sensor.supports(rs.option.depth_offset):
            depth_offset_meters = depth_sensor.get_option(rs.option.depth_offset)
            depth_offset_mm = depth_offset_meters * 1000
            if abs(depth_offset_mm) > 1:
                print(f"\n⚠️  DEPTH OFFSET DETECTED: {depth_offset_mm:.1f}mm will be automatically subtracted")
        
        print("\n" + "=" * 70)
        print("LIVE CAPTURE")
        print("=" * 70)
        print("Press 'q' to quit")
        print("Press 's' to save sample (RGB, depth, aligned)")
        print("Press 'd' to toggle depth visualization mode")
        print("Click on image to see depth value at that point")
        if abs(depth_offset_mm) > 1:
            print(f"NOTE: Compensating for {depth_offset_mm:.1f}mm depth offset automatically")
        print("=" * 70)
        
        frame_count = 0
        depth_viz_mode = 0  # 0=auto, 1=fixed 0-2m, 2=fixed 0-1m
        click_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_point
            if event == cv2.EVENT_LBUTTONDOWN:
                click_point = (x, y)
        
        cv2.namedWindow('RealSense L515: RGB | Aligned Depth')
        cv2.setMouseCallback('RealSense L515: RGB | Aligned Depth', mouse_callback)
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = align_to_color.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not aligned_depth_frame:
                continue
            
            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # CORRECT METHOD: Use get_distance() for a sample, but vectorize for speed
            # The SDK's get_distance() handles the offset automatically
            # We'll use it for display but need faster method for full array
            
            # For full array: use the raw data but apply proper formula
            # According to RealSense docs: actual_distance = raw_value * depth_scale
            # The depth_scale already accounts for offset in L515
            depth_data_raw = np.asanyarray(aligned_depth_frame.get_data())
            
            # Use get_distance() for center point to verify
            h, w = depth_data_raw.shape
            center_dist_api = aligned_depth_frame.get_distance(w//2, h//2) * 1000  # mm
            
            # Try the simple conversion
            aligned_depth_image = depth_data_raw.astype(np.float32) * depth_scale * 1000  # to mm
            center_dist_calc = aligned_depth_image[h//2, w//2]
            
            # Check if they match - if not, offset compensation is needed
            if abs(center_dist_api - center_dist_calc) > 10:  # More than 10mm difference
                # Offset not handled by depth_scale, subtract manually
                aligned_depth_image = aligned_depth_image - depth_offset_mm
                aligned_depth_image[aligned_depth_image < 0] = 0
            
            # Now both are 640x480!
            assert color_image.shape[:2] == aligned_depth_image.shape[:2], \
                "After alignment, shapes should match!"
            
            # Get depth statistics (in mm)
            valid_depth = aligned_depth_image[aligned_depth_image > 0]
            if len(valid_depth) > 0:
                depth_min = np.min(valid_depth)
                depth_max = np.max(valid_depth)
                depth_mean = np.mean(valid_depth)
            else:
                depth_min = depth_max = depth_mean = 0
            
            # Colorize depth based on mode
            if depth_viz_mode == 0:  # Auto
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
            elif depth_viz_mode == 1:  # Fixed 0-2m
                depth_normalized = np.clip(aligned_depth_image / 2000.0, 0, 1) * 255
                depth_colormap = cv2.applyColorMap(
                    depth_normalized.astype(np.uint8),
                    cv2.COLORMAP_JET
                )
            else:  # Fixed 0-1m
                depth_normalized = np.clip(aligned_depth_image / 1000.0, 0, 1) * 255
                depth_colormap = cv2.applyColorMap(
                    depth_normalized.astype(np.uint8),
                    cv2.COLORMAP_JET
                )
            
            # Handle click point
            if click_point is not None:
                cx, cy = click_point
                # Adjust for side-by-side display
                if cx < 640:  # Clicked on RGB side
                    depth_at_click = aligned_depth_image[cy, cx]
                    cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), 2)
                    cv2.putText(color_image, f"{depth_at_click}mm ({depth_at_click/10:.1f}cm)", 
                               (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:  # Clicked on depth side
                    dx = cx - 640
                    depth_at_click = aligned_depth_image[cy, dx]
                    cv2.circle(depth_colormap, (dx, cy), 5, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, f"{depth_at_click}mm", 
                               (dx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Stack images
            images = np.hstack((color_image, depth_colormap))
            
            # Add text overlay
            viz_modes = ["Auto", "0-2m", "0-1m"]
            cv2.putText(images, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(images, f"Depth Range: {depth_min}mm - {depth_max}mm (avg: {depth_mean:.0f}mm)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(images, f"Viz Mode: {viz_modes[depth_viz_mode]} (press 'd')", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(images, f"Min: {depth_min/10:.1f}cm, Max: {depth_max/10:.1f}cm", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('RealSense L515: RGB | Aligned Depth', images)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                # Save samples
                cv2.imwrite('l515_rgb.png', color_image)
                cv2.imwrite('l515_depth_colormap.png', depth_colormap)
                # Save as uint16 in mm (after offset correction)
                np.save('l515_depth_aligned.npy', aligned_depth_image.astype(np.uint16))
                
                # Also save intrinsics and extrinsics
                camera_params = {
                    'color_intrinsics': {
                        'width': color_intrinsics.width,
                        'height': color_intrinsics.height,
                        'fx': color_intrinsics.fx,
                        'fy': color_intrinsics.fy,
                        'cx': color_intrinsics.ppx,
                        'cy': color_intrinsics.ppy,
                        'coeffs': list(color_intrinsics.coeffs)
                    },
                    'depth_scale': depth_scale,
                    'depth_offset_mm': depth_offset_mm,
                    'depth_units': 'millimeters',
                    'note': f'Depth values in mm, offset of {depth_offset_mm:.1f}mm already subtracted',
                    'depth_to_color_extrinsics': {
                        'rotation': rotation.tolist() if 'rotation' in locals() else None,
                        'translation': list(extrinsics.translation) if 'extrinsics' in locals() else None
                    }
                }
                np.save('l515_camera_params.npy', camera_params)
                
                print(f"\n✓ Saved at frame {frame_count}:")
                print(f"  - l515_rgb.png")
                print(f"  - l515_depth_colormap.png")
                print(f"  - l515_depth_aligned.npy (shape: {aligned_depth_image.shape}, dtype: uint16, in mm)")
                print(f"  - l515_camera_params.npy")
                
            elif key & 0xFF == ord('d'):
                depth_viz_mode = (depth_viz_mode + 1) % 3
                print(f"Depth visualization mode: {viz_modes[depth_viz_mode]}")
                
            frame_count += 1
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline_started:
            pipeline.stop()
            print(f"\n✓ Captured {frame_count} frames. Pipeline stopped.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test_l515_advanced()
