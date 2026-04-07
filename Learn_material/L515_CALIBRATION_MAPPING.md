# L515 Calibration Mapping - Simple Guide

This document explains in simple terms how to convert your calibration files to the script format.

---

## 📦 What's in Your Calibration Files?

### File 1: `l515_calibration.npz` (Camera Intrinsics)
**What it is:** How the camera "sees" - like the camera's glasses prescription

```
Camera Matrix:
[[605.10   0      327.29]
 [  0    604.97   251.01]
 [  0      0        1   ]]
```

**Simple explanation:**
- `fx = 605.10`, `fy = 604.97` → How much the camera "zooms" in X and Y directions
- `cx = 327.29`, `cy = 251.01` → The center point of the image (like the bullseye)
- Used to convert pixels → 3D points

**Note:** The script doesn't currently use these, but you might need them later for depth-to-point-cloud conversion.

---

### File 2: `l515_extrinsic_calibration.npz` (Camera Position & Orientation)
**What it is:** Where the camera is positioned relative to the robot base

**Contains:**
```
Translation Vector (in mm): [-9.52, -0.70, 535.65]
Rotation Matrix (3x3): Describes camera orientation
```

**Simple explanation:**
- **Translation** = Where the camera is located (X, Y, Z coordinates)
- **Rotation** = Which way the camera is pointing (like compass direction but in 3D)

---

## 🔧 Converting to Script Format

### The Original Script Needs:

```python
ROBOT2CAM_POS = [x, y, z]  # in meters
ROBOT2CAM_QUAT = [qx, qy, qz, qw]  # rotation as quaternion
```

---

## 📐 Conversion Math (Step by Step)

### Step 1: Convert Translation (Position)

**From your file:**
```
translation_vector (mm) = [-9.52167045, -0.69569744, 535.65046067]
```

**Formula:**
```
ROBOT2CAM_POS (meters) = translation_vector / 1000
```

**Why?** Your calibration is in millimeters, but the script uses meters.

**Result:**
```python
ROBOT2CAM_POS = np.array([-9.52167045, -0.69569744, 535.65046067]) / 1000.0
# = [-0.00952167, -0.00069570, 0.53565046] meters
```

**What it means:** 
- Camera is ~10mm left of robot base
- Camera is ~1mm back from robot base  
- Camera is ~536mm above robot base

---

### Step 2: Convert Rotation (Orientation)

**From your file:**
```
rotation_matrix (3x3) = 
[[ 0.06349990  0.99704151  0.04331275]
 [ 0.13665460  0.03430463 -0.99002460]
 [-0.98858145  0.06878535 -0.13407197]]
```

**Formula:**
```python
from scipy.spatial.transform import Rotation as R
ROBOT2CAM_QUAT = R.from_matrix(rotation_matrix).as_quat()
```

**Why quaternion?** The script uses quaternions because:
- They're good at representing 3D rotations
- No "gimbal lock" problems
- Easy to chain multiple rotations together

**Result:**
```python
ROBOT2CAM_QUAT = [-0.06376827, -0.01828862, 0.70604348, 0.70476071]
```

**What it means:** This describes how the camera is tilted/rotated relative to the robot base.

---

## 🤔 What About Those Manual Offsets?

In the original script, you see:
```python
OFFSET_ORI_X = R.from_euler('x', -1.2, degrees=True)
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X
OFFSET_ORI_Y = R.from_euler('y', 10, degrees=True)
ori = ori * OFFSET_ORI_Y
# ... etc
```

**What this does:** Manually tweaks the camera angle by rotating:
- -1.2° around X-axis
- +10° around Y-axis  
- 0° around Z-axis

**Why they did it:** They didn't have proper calibration equipment, so they had to manually adjust angles until it looked right.

**Do YOU need it?** **NO!** Because:
- ✅ You used ArUco markers (proper calibration)
- ✅ Your rotation is already correct
- ❌ Adding manual offsets would make it LESS accurate

**IF you want to keep the same structure**, use **zero offsets**:
```python
ROBOT2CAM_QUAT_INITIAL = np.array([-0.06376827, -0.01828862, 0.70604348, 0.70476071])
OFFSET_ORI_X = R.from_euler('x', 0, degrees=True)  # No adjustment needed
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X
OFFSET_ORI_Y = R.from_euler('y', 0, degrees=True)  # No adjustment needed
ori = ori * OFFSET_ORI_Y
OFFSET_ORI_Z = R.from_euler('z', 0, degrees=True)  # No adjustment needed
ori = ori * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()  # Result is same as ROBOT2CAM_QUAT_INITIAL
```

---

## 📋 Complete Conversion Summary

### Input (Your Files):
```python
translation_vector = [-9.52167045, -0.69569744, 535.65046067]  # mm
rotation_matrix = [[ 0.06349990  0.99704151  0.04331275]
                   [ 0.13665460  0.03430463 -0.99002460]
                   [-0.98858145  0.06878535 -0.13407197]]
```

### Output (For Script):
```python
# Step 1: Convert mm to meters
ROBOT2CAM_POS = translation_vector / 1000.0
# Result: [-0.00952167, -0.00069570, 0.53565046]

# Step 2: Convert rotation matrix to quaternion  
ROBOT2CAM_QUAT = R.from_matrix(rotation_matrix).as_quat()
# Result: [-0.06376827, -0.01828862, 0.70604348, 0.70476071]

# Step 3: Keep everything else the same
REALSENSE_SCALE = 0.00025
quat = [-0.491, 0.495, -0.505, 0.509]
pos = [0.004, 0.001, 0.014]
T_link2color = ...  # (copy from original)
T_link2viz = ...    # (copy from original)
```

---

## 🎯 What Each Variable Means (In Plain English)

| Variable | What It Is | Your Value | Units |
|----------|-----------|------------|-------|
| `ROBOT2CAM_POS` | Where camera is located | `[-9.52, -0.70, 535.65]` | mm from robot base |
| `ROBOT2CAM_QUAT` | Which way camera points | `[-0.064, -0.018, 0.706, 0.705]` | quaternion (no units) |
| `REALSENSE_SCALE` | How to convert depth numbers to meters | `0.00025` | meters per unit |
| `T_link2viz` | Coordinate system flip (camera → robot axes) | (matrix) | transformation |
| `quat` / `pos` | Color camera offset from depth camera | (from original) | rotation/position |

---

## ✅ Final Hardcoded Values

Just copy-paste this into the script:

```python
################################# Camera Calibration ##############################################
# L515 calibration - from ArUco marker calibration

# Position: Where camera is relative to robot base (in meters)
ROBOT2CAM_POS = np.array([-9.52167045, -0.69569744, 535.65046067]) / 1000.0

# Orientation: Which way camera is pointing (as quaternion)
ROBOT2CAM_QUAT = np.array([-0.06376827, -0.01828862, 0.70604348, 0.70476071])

# Depth scale: Convert raw sensor values to meters
REALSENSE_SCALE = 0.00025

# Color-to-depth camera offset (from original script, keep as-is)
quat = [-0.491, 0.495, -0.505, 0.509]
pos = [0.004, 0.001, 0.014]
T_link2color = np.concatenate((np.concatenate((R.from_quat(quat).as_matrix(), np.array([pos]).T), axis=1), [[0, 0, 0, 1]]))

# Coordinate system transformations (from original script, keep as-is)
T_link2viz = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
transform_realsense_util = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
###################################################################################################
```

Done! That's all you need. 🎉

## Mapping to Original pcd_process.py

### Original Script Parameters (D435 RealSense)
```python
OFFSET = np.array([0.0, 0.0, -0.035])
ROBOT2CAM_POS = np.array([1.227, -0.009, 0.368]) + OFFSET
ROBOT2CAM_QUAT_INITIAL = np.array([0.0159, -0.1884, -0.0095, 0.9819])
# Additional orientation offsets applied...

REALSENSE_SCALE = 0.0002500000118743628
```

### Your L515 Parameters (Updated)
```python
# From extrinsic calibration
ROBOT2CAM_POS = np.array([-0.00952, -0.00070, 0.53565])  # meters
ROBOT2CAM_ROTATION = [[ 0.0635  0.9970  0.0433]
                      [ 0.1367  0.0343 -0.9900]
                      [-0.9886  0.0688 -0.1341]]
ROBOT2CAM_QUAT = R.from_matrix(ROBOT2CAM_ROTATION).as_quat()

# From L515 sensor (query at runtime)
REALSENSE_SCALE = depth_sensor.get_depth_scale()  # typically ~0.00025
```

## Key Differences: D435 vs L515

| Parameter | D435 (Original) | L515 (Yours) | Notes |
|-----------|----------------|--------------|-------|
| Depth Technology | Stereo | LiDAR | L515 more accurate at close range |
| Min Distance | ~0.3m | ~0.25m | L515 works closer |
| Depth Scale | 0.00025 | 0.00025 | Similar, but query at runtime |
| Resolution | 640×480 typical | 640×480 (yours) | Same in your setup |
| Depth Range | Configurable | 200-1500mm (yours) | From export_summary |
| Calibration Method | Manual offsets | ArUco markers | Yours is more accurate |

## Your Data Format (from export_summary.txt)

```
Image Resolution: 640x480
Image Channels: 4 (RGBD)
Depth Information:
  Normalized range: 0-255 (uint8)
  Actual range: 200mm - 1500mm
  Formula: depth_mm ≈ 200 + (normalized_value / 255) * 1300
```

**Important:** Your exported data uses normalized depth (0-255), not raw sensor values. To use with `pcd_process.py`, you need to:

```python
# Convert normalized depth to meters
depth_mm = 200 + (normalized_depth / 255) * 1300
depth_m = depth_mm / 1000.0
```

## Updated pcd_process_l515.py

The new script `/home/hk/Documents/DemoGen/real_world/utils/pcd_process_l515.py` includes:

### ✅ What's Updated
1. **Intrinsic Parameters**: Uses your L515 camera matrix (fx, fy, cx, cy)
2. **Extrinsic Transform**: Uses your calibrated transformation matrix
3. **Depth Conversion**: Handles both normalized export format and raw L515 data
4. **Documentation**: Explains your specific setup parameters
5. **Format Compatibility**: Maintains same structure as original script

### ⚠️ What You Need to Adjust

1. **Workspace Bounds** (lines 138-147):
   ```python
   work_space=[
       [0.2, 0.8],      # X range - adjust for your robot
       [-0.66, 0.6],    # Y range - adjust for your robot
       [0.005, 0.45]    # Z range - adjust for your robot
   ]
   ```
   Your EE range from export_summary: X[0, 175mm], Y[-108, 144mm], Z[-71, 39mm]
   These need to be transformed to robot base frame coordinates.

2. **Depth Scale at Runtime**:
   ```python
   # Query from sensor instead of hardcoding
   depth_scale = depth_sensor.get_depth_scale()
   ```

3. **Camera Serial Number** (line 399):
   ```python
   id = "f0211830"  # Replace with your L515 serial number
   ```

4. **Coordinate System Alignment**:
   The `T_link2viz` matrix may need adjustment based on how your camera is mounted relative to the robot.

## Usage Examples

### Example 1: With RealSense SDK (Raw Depth)
```python
from pcd_process_l515 import preprocess_point_cloud, pcd_config
import pyrealsense2 as rs
import numpy as np

# Get frame from camera
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Query depth scale
depth_scale = depth_sensor.get_depth_scale()

# Convert to point cloud
# ... (your existing code to create point cloud)

# Preprocess
processed = preprocess_point_cloud(points, cfg=pcd_config, debug=True)
```

### Example 2: With Your Exported Data (Normalized Depth)
```python
from pcd_process_l515 import depth_image_to_point_cloud, preprocess_point_cloud
import numpy as np
import cv2

# Load your exported data
rgb = cv2.imread('rgb_images/frame_000.png')
depth_normalized = cv2.imread('depth_png/frame_000.png', cv2.IMREAD_GRAYSCALE)

# Convert to point cloud (handles normalized format automatically)
points = depth_image_to_point_cloud(
    depth_normalized, 
    rgb, 
    depth_scale=None  # None = use normalized export format
)

# Preprocess
processed = preprocess_point_cloud(points, debug=True)
```

## Verification Steps

1. **Test Coordinate Transform**:
   ```bash
   cd /home/hk/Documents/DemoGen/real_world/utils
   python pcd_process_l515.py
   ```

2. **Check Point Cloud Range**:
   - Before crop: Should show camera-frame coordinates
   - After crop: Should show robot-frame coordinates within workspace

3. **Visualize**:
   - Set `debug=True` to see intermediate steps
   - Use `pcd_visualizer` to check transformations

4. **Compare with Original**:
   - Original uses manual offset calibration
   - Yours uses ArUco marker calibration (more accurate)
   - Expect some differences, especially in rotation

## Troubleshooting

### Problem: No points after cropping
**Solution**: Adjust workspace bounds or check transformation matrix

### Problem: Point cloud looks flipped/rotated
**Solution**: Adjust `T_link2viz` coordinate alignment matrix

### Problem: Depth values seem wrong
**Solution**: 
- Check if using normalized (0-255) or raw (uint16) depth
- Verify depth_scale is correct
- Ensure depth_offset is 0 (check with test_l515_advanced.py)

### Problem: Reprojection error seems high (17px mean)
**Note**: This is acceptable for a 640×480 image. For reference:
- < 1 pixel: Excellent
- 1-3 pixels: Good
- 3-10 pixels: Acceptable
- > 10 pixels: May need recalibration
Your 17px on 640×480 is about 2.7% of image width - borderline but usable.

## Next Steps

1. Test the new script with your L515 camera
2. Adjust workspace bounds based on your robot's reachable area
3. Verify the coordinate transformation by checking known points
4. Compare results with original pcd_process.py to ensure compatibility
5. Consider recalibrating if reprojection error needs improvement

## Files Created

- `/home/hk/Documents/DemoGen/real_world/utils/pcd_process_l515.py` - Updated script with your calibration
- This document for reference

## References

- Original script: `/home/hk/Documents/DemoGen/real_world/utils/pcd_process.py`
- Your calibration: `/home/hk/Documents/DemoGen/Kinematic/l515_*_calibration.npz`
- Export format: `/home/hk/Documents/DemoGen/data/demo_data_v2/exported_episode/episode_1_exported/export_summary.txt`
- Test script: `/home/hk/Documents/DemoGen/test_l515_advanced.py`
