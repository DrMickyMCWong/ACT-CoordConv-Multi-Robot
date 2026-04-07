#!/usr/bin/env python3
"""
Script to display live top camera feed
"""
import sys
import os
sys.path.append('/home/hk/Documents/ACT_Shaka/act-main')

import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from act.record_sim_TCP_EE_depth_cam_scripted import capture_image_usb

def display_top_camera_live():
    """Display live feed from the top USB camera"""
    print("Starting top camera live display...")
    print("Press 'q' to quit")
    
    try:
        # Create window
        cv2.namedWindow('Top Camera Live', cv2.WINDOW_AUTOSIZE)
        
        while True:
            # Capture image from USB camera
            image = capture_image_usb(0)  # Use device 0 (USB camera)
            
            if image is not None and image.size > 0:
                # Convert RGB to BGR for OpenCV display
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Display image
                cv2.imshow('Top Camera Live', image_bgr)
                
                # Check for 'q' key press
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
            else:
                print("Failed to capture image from top camera")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Live display closed")

def display_top_camera_from_episode():
    """Display top camera images from the recorded episode"""
    episode_path = "/home/hk/Documents/ACT_Shaka/act-main/act/dataset_dir/real_episodes/episode_0.hdf5"
    
    if not Path(episode_path).exists():
        print(f"Episode file not found: {episode_path}")
        return
    
    try:
        with h5py.File(episode_path, 'r') as f:
            print("Available datasets in episode:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape} ({obj.dtype})")
                else:
                    print(f"  {name}: (group)")
            
            f.visititems(print_structure)
            
            if 'observations' in f and 'images' in f['observations']:
                images = f['observations']['images']
                print("\nAvailable camera views:")
                for cam_name in images.keys():
                    if isinstance(images[cam_name], h5py.Dataset):
                        print(f"  {cam_name}: {images[cam_name].shape}")
                
                if 'top' in images:
                    top_images = images['top'][:]
                    print(f"\nTop camera data shape: {top_images.shape}")
                    print(f"Data type: {top_images.dtype}")
                    print(f"Value range: {top_images.min()} - {top_images.max()}")
                    
                    # Display first few frames
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle("Top Camera Images (First 6 Frames)")
                    
                    for i in range(min(6, len(top_images))):
                        row = i // 3
                        col = i % 3
                        
                        img = top_images[i]
                        if img.dtype == np.uint8:
                            # Image is already in correct format
                            display_img = img
                        else:
                            # Normalize to 0-255 range
                            display_img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        
                        axes[row, col].imshow(display_img)
                        axes[row, col].set_title(f"Frame {i}")
                        axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Also save first frame as image file
                    first_frame = top_images[0]
                    if first_frame.dtype != np.uint8:
                        first_frame = ((first_frame - first_frame.min()) / (first_frame.max() - first_frame.min()) * 255).astype(np.uint8)
                    
                    output_path = "/home/hk/Documents/ACT_Shaka/top_camera_sample.jpg"
                    cv2.imwrite(output_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
                    print(f"\nFirst frame saved to: {output_path}")
                    
                else:
                    print("\nNo 'top' camera found in recorded data!")
            else:
                print("\nNo image data found in episode!")
                
    except Exception as e:
        print(f"Error reading episode file: {e}")

def test_live_top_camera():
    """Test live top camera feed"""
    print("\n" + "="*50)
    print("Testing Live Top Camera Feed")
    print("="*50)
    
    try:
        # Try to open USB camera at /dev/video8
        cap = cv2.VideoCapture(8)
        
        if not cap.isOpened():
            print("Failed to open camera at /dev/video8")
            return
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Top camera opened successfully!")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Display frame
            cv2.imshow('Top Camera Live Feed', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"/home/hk/Documents/ACT_Shaka/top_camera_live_{frame_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Frame saved to: {save_path}")
                frame_count += 1
                
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error with live camera: {e}")

if __name__ == "__main__":
    print("Top Camera Display Test")
    print("="*50)
    
    print("\n1. Testing live camera feed...")
    response = input("Do you want to test live top camera? (y/n): ")
    if response.lower() == 'y':
        display_top_camera_live()
    
    print("\n2. Checking recorded episode data...")
    display_top_camera_from_episode()
    
    print("\nDone!")