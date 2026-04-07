import cv2
import numpy as np

print("🎥 Dual Camera Test")
print("Press ESC to exit")

# Open both cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if cap0.isOpened() and cap1.isOpened():
    print("✅ Both cameras opened successfully!")
    
    while True:
        # Capture from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("❌ Failed to capture from one or both cameras")
            break
        
        # Resize frames to same size if needed
        height = min(frame0.shape[0], frame1.shape[0])
        width = min(frame0.shape[1], frame1.shape[1])
        
        frame0_resized = cv2.resize(frame0, (width, height))
        frame1_resized = cv2.resize(frame1, (width, height))
        
        # Add labels
        cv2.putText(frame0_resized, "Camera 0", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame1_resized, "Camera 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine frames side by side
        combined = np.hstack((frame0_resized, frame1_resized))
        
        # Add instruction
        cv2.putText(combined, "Press ESC to exit", 
                   (10, combined.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display combined view
        cv2.imshow('Dual Camera View', combined)
        
        # Check for ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print("ESC pressed - closing cameras")
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
else:
    print("❌ Cannot open one or both cameras")
    if cap0.isOpened():
        cap0.release()
    if cap1.isOpened():
        cap1.release()