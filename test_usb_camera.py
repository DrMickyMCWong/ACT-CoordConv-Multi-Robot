#!/usr/bin/env python3
import cv2
import numpy as np

def test_camera_device(device_id):
    print(f"Testing camera device {device_id}...")
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Failed to open camera {device_id}")
            return False

        # Force target resolution for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Failed to capture frame from camera {device_id}")
            cap.release()
            return False
        
        print(f"Successfully captured frame from camera {device_id}: {frame.shape}")
        
        # Check frame properties
        height, width = frame.shape[:2]
        print(f"Frame size: {width}x{height}")
        print(f"Frame dtype: {frame.dtype}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"Exception testing camera {device_id}: {e}")
        return False

def live_stream_camera(device_id=0):
    """Live stream from specified camera device"""
    print(f"\nStarting live stream from camera {device_id}...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Failed to open camera {device_id} for streaming")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Add frame info overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow(f'USB Camera {device_id} Live Stream', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting live stream...")
                break
            elif key == ord('s'):
                resized = cv2.resize(frame, (640, 480))
                filename = f"camera_{device_id}_frame_{frame_count}.png"
                cv2.imwrite(filename, resized)
                print(f"Saved frame as {filename} (640x480 PNG)")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Live stream ended. Total frames: {frame_count}")
        
    except Exception as e:
        print(f"Exception during live streaming: {e}")
        cv2.destroyAllWindows()

def main():
    # Test video devices from 0 to 10
    working_devices = []
    
    for device_id in range(11):
        device_path = f"/dev/video{device_id}"
        print(f"\n--- Testing {device_path} ---")
        
        if test_camera_device(device_id):
            working_devices.append(device_id)
        
    print(f"\nWorking camera devices: {working_devices}")
    
    # If /dev/video0 is working, offer live streaming
    if 0 in working_devices:
        print("\n--- Live Stream Option ---")
        response = input("Do you want to start live streaming from /dev/video0? (y/n): ")
        if response.lower() in ['y', 'yes']:
            live_stream_camera(0)
    else:
        print("\n/dev/video0 is not available for live streaming")

if __name__ == "__main__":
    main()