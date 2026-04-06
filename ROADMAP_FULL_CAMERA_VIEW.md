# Roadmap: Transitioning from Cropped to Full Camera View

## Current Situation

### task7c (Original - Working at 90%)
- **Training data**: Cropped (90,0) to (600,480), then resized to 640x480
- **Evaluation**: Same cropping applied
- **What's lost**: Left 90 pixels, right 680 pixels of 1280 width camera view
- **Effective field of view**: Only middle 510 pixels → resized to 640

### task7c_new (Newly Trained - NOT TESTED YET)
- **Training data**: Full 720x1280 image → resized to 640x480 (NO cropping)
- **Evaluation**: Currently using cropping (MISMATCH!)
- **Status**: Need to evaluate WITHOUT cropping to match training

---

## Why Use Full Camera View?

### Advantages ✅
1. **More information**: Wider field of view = more context
2. **Less preprocessing**: Simpler pipeline (just resize)
3. **Better peripheral awareness**: Can see objects at edges
4. **More robust**: Less sensitive to camera positioning

### Potential Issues ⚠️
1. **Aspect ratio change**: 1280/720 = 1.78 vs 640/480 = 1.33 (distortion when resizing)
2. **Lower effective resolution**: Spreading 1280 pixels to 640 = 50% horizontal compression
3. **More irrelevant info**: May include workspace edges, walls, etc.

---

## Roadmap: Testing Full Camera View

### Phase 1: Verify What You Have ✅ (Do This First)

#### Step 1.1: Check Raw Recording Resolution
```bash
cd /home/hk/Documents/ACT_Shaka
python3 -c "
import cv2
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
print(f'Camera resolution: {frame.shape}')
cam.release()
"
```

**Expected output**: Something like `(720, 1280, 3)` or `(1080, 1920, 3)`

#### Step 1.2: Visualize Cropped vs Full View
```bash
python3 << 'EOF'
import cv2
import numpy as np

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

# Show what's currently cropped
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
x1, y1, x2, y2 = 90, 0, 600, 480

# Draw rectangle showing crop area
frame_with_box = frame.copy()
cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(frame_with_box, "GREEN = Used in training", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Create cropped version
cropped = frame_rgb[y1:y2, x1:x2]
cropped_resized = cv2.resize(cropped, (640, 480))

# Create full version
full_resized = cv2.resize(frame_rgb, (640, 480))

# Save comparison
cv2.imwrite('camera_full_view.jpg', frame_with_box)
cv2.imwrite('camera_cropped_used.jpg', cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR))
cv2.imwrite('camera_full_resized.jpg', cv2.cvtColor(full_resized, cv2.COLOR_RGB2BGR))

print("✅ Saved comparison images:")
print("   - camera_full_view.jpg (shows what's cropped out)")
print("   - camera_cropped_used.jpg (current training view)")
print("   - camera_full_resized.jpg (proposed full view)")

cam.release()
EOF
```

**Action**: Look at these images and decide if the cropped parts contain useful information!

---

### Phase 2: Test task7c_new WITHOUT Cropping ⚡ (Quick Test)

#### Step 2.1: Update Config for task7c_new
```bash
# Edit config/config.py
# Change:
'eval_ckpt_name': 'policy_best_epoch_5303_val_0.0567.ckpt',
```

#### Step 2.2: Update evaluate_custom.py to NOT crop
```python
def capture_image(cam):
    """Capture full camera view - NO cropping"""
    _, frame = cam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # NO CROPPING - matches task7c_new training data
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    return image
```

#### Step 2.3: Run Evaluation
```bash
python evaluate_custom.py --task task7c_new
```

**Expected Results**:
- ✅ **If it works well**: Full camera view is usable! Continue with it.
- ❌ **If it fails**: Need to re-record OR re-crop training data

---

### Phase 3: Decision Matrix

| Scenario | Action | Effort |
|----------|--------|--------|
| **task7c_new works well with full view** | ✅ Use it! No cropping needed | LOW - You're done! |
| **task7c_new doesn't work** | Option A: Re-record 90 episodes with cropping | HIGH - 1-2 hours |
| | Option B: Re-convert data with cropping | MEDIUM - 30 mins + retrain |
| | Option C: Use task7c (cropped) for now | ZERO - Keep current |

---

### Phase 4A: Re-Convert Data WITH Cropping (If Needed)

If you decide full view doesn't help and want cropping:

```python
# Create: convert_episodes_with_crop.py

def resize_and_crop_image(image, target_height=480, target_width=640):
    """Crop then resize - matches task7c training"""
    # Crop first
    x1, y1, x2, y2 = 90, 0, 600, 480
    cropped = image[:, x1:x2, :]  # Note: images are (H, W, C) in HDF5
    # Then resize
    return cv2.resize(cropped, (target_width, target_height), 
                     interpolation=cv2.INTER_LINEAR)
```

Then re-run conversion and retrain.

---

### Phase 4B: Record New Data with Better Camera Setup

If you want to optimize further:

1. **Adjust camera position**: Frame the task area better
2. **Use higher resolution**: If camera supports it
3. **Better lighting**: Improves model performance
4. **Consistent background**: Remove clutter

---

## Quick Reference: What Changes Based on Cropping

### WITH Cropping (task7c - current working)
```python
# evaluate_custom.py
image = image[0:480, 90:600]  # Crop
image = cv2.resize(image, (640, 480))  # Resize

# Training data
# Already cropped in original recordings
```

### WITHOUT Cropping (task7c_new - your new training)
```python
# evaluate_custom.py  
# No cropping!
image = cv2.resize(image, (640, 480))  # Just resize

# Training data
# Raw 720x1280 → resized to 480x640
```

---

## Recommended Next Steps

1. ✅ **Run Phase 1 visualization** - See what you're missing
2. ✅ **Test task7c_new without cropping** - 5 minute test
3. ⏸️ **Decide based on results**:
   - Works well → Keep full view, update docs
   - Doesn't work → Re-convert with cropping OR stick with task7c

---

## Terminal Output Cleanup (Bonus)

The verbose warnings you see can be suppressed:

```python
# Add to top of evaluate_custom.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Or set environment variable
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
```

---

## Summary

**Immediate Action**: 
1. Run Phase 1 visualization to see if cropping removes useful info
2. Update evaluate_custom.py to remove cropping
3. Test task7c_new model
4. Report back: Does it work? Better or worse than task7c?

The answer will tell us whether to embrace full camera view or stick with cropping.
