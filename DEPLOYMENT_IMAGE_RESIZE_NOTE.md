# Important Note: Image Resizing for Model Deployment

## ⚠️ Critical for Deployment

Your trained model will expect **480x640** resolution images, but your camera captures at **720x1280**.

### What This Means

1. **During Training**: Images are automatically resized from 720x1280 → 480x640
2. **During Deployment**: You MUST resize camera input to 480x640 before feeding to the model

### Implementation

Check your evaluation/deployment scripts to ensure they resize images correctly:

```python
import cv2

# When capturing from camera
camera_image = capture_from_camera()  # Shape: (720, 1280, 3)

# Resize to match training data
model_input = cv2.resize(camera_image, (640, 480), interpolation=cv2.INTER_LINEAR)
# Shape: (480, 640, 3)

# Now feed to model
prediction = model(model_input)
```

### Files to Check

Look for image preprocessing in these files:
- `evaluate_custom.py`
- `evaluate_custom_two_cam.py`
- Any teleoperation or deployment scripts

The `training/utils.py` `get_image()` function already handles this for offline evaluation, but real-time deployment scripts may need updates.

### Why This Matters

- **Mismatch = Poor Performance**: If you feed 720x1280 images to a model trained on 480x640, it will fail
- **Aspect Ratio**: Both resolutions have the same 4:3 aspect ratio (good!)
- **Information Loss**: Minimal - downsampling from higher resolution is generally fine

### Quick Check

In your deployment script, add this assertion:
```python
assert image.shape == (480, 640, 3), f"Expected (480, 640, 3), got {image.shape}"
```

This will catch size mismatches immediately during testing.
