# Data Collection Guide for DemoGen

Complete guide for collecting your own demonstration data compatible with DemoGen.

## 📋 Overview

Based on analysis of the source datasets and existing collection scripts, here's what you need to collect demonstration data for DemoGen.

---

## 🎯 Required Hardware

### 1. **RGB-D Camera**
- **Recommended:** Intel RealSense L515 or D435
- **Used in source data:** L515 (ID: `f0211830`)
- **Requirements:**
  - Depth resolution: 640x480 or higher
  - RGB resolution: 640x480 or higher
  - Aligned RGB-D frames

### 2. **Robot Arm**
- **Used in source data:** Franka Emika Panda (via Polymetis)
- **Requirements:**
  - Cartesian pose feedback (x, y, z, roll, pitch, yaw)
  - Gripper state feedback
  - Real-time control interface

### 3. **Gripper/Hand**
- **Used in source data:** OYhand (multi-finger dexterous hand)
- **Requirements:**
  - Binary open/close state (minimum)
  - Position feedback

---

## 📊 Data Format Requirements

### Data Structure (zarr format)
```
your_task.zarr/
├── data/
│   ├── action (T, 7)          # Delta movements
│   ├── state (T, 7)           # Absolute poses
│   └── point_cloud (T, 1024, 6)  # RGB-D observations
└── meta/
    └── episode_ends (n_episodes,)  # Episode boundaries
```

### Per-Timestep Data

#### 1. **state** `(7,)` - Absolute Robot Pose
```python
state = [x, y, z, euler_x, euler_y, euler_z, gripper]
```
- **Position (XYZ):** End-effector position in robot base frame (meters)
- **Orientation (Euler):** Roll, Pitch, Yaw in radians
- **Gripper:** -1.0 (open) to 1.0 (closed)
- **Normalized to:** [-1.0, 1.5] range (workspace-dependent)

**Example from flower.zarr:**
```
Frame 0: [ 0.505,  0.065,  0.280,  0.000,  0.000,  0.000, -1.000]
Frame 1: [ 0.490,  0.080,  0.280,  0.000,  0.000,  0.000, -1.000]
```

#### 2. **action** `(7,)` - Delta Movements
```python
action = state[t+1] - state[t]  # Incremental command
```
- **Format:** Same as state but represents change between frames
- **Normalized to:** [-1.0, 1.0] range
- **Typical values:** ±0.015 for position, 0-0.1 for orientation

**Example from flower.zarr:**
```
Frame 0: [-0.015,  0.015,  0.000,  0.000,  0.000,  0.000, -1.000]
```

#### 3. **point_cloud** `(1024, 6)` - RGB-D Observation
```python
point_cloud = [
    [x1, y1, z1, r1, g1, b1],
    [x2, y2, z2, r2, g2, b2],
    ...
    [x1024, y1024, z1024, r1024, g1024, b1024]
]
```
- **XYZ:** 3D coordinates in robot base frame (meters)
- **RGB:** Color values in range [0, 255]
- **Downsampled to:** 1024 points (from original dense cloud)
- **Frame:** Robot base frame (NOT camera frame!)

---

## 🔧 Data Collection Pipeline

### Step 1: Camera Setup & Calibration

```python
from realsense_camera import RealSense_Camera

# Initialize camera
camera = RealSense_Camera(type="L515", id="YOUR_CAMERA_ID")
camera.prepare()

# Get raw data
point_cloud, rgbd_frame = camera.get_frame()
# point_cloud: (H*W, 3) XYZ in camera frame
# rgbd_frame: (H, W, 4) RGBD image
```

**Camera-to-Robot Transform:**
You need to calibrate the transformation from camera frame to robot base frame. This is typically done with:
- Hand-eye calibration (camera on robot) OR
- Fixed camera calibration (camera in workspace)

### Step 2: Point Cloud Processing

```python
from pcd_process import preprocess_point_cloud, pcd_crop, pcd_cluster

# 1. Transform to robot base frame
point_cloud_robot = transform_camera_to_robot(point_cloud, camera_pose)

# 2. Crop to workspace
point_cloud_cropped = pcd_crop(point_cloud_robot)

# 3. Downsample to 1024 points
point_cloud_sampled = downsample_fps(point_cloud_cropped, n_points=1024)

# 4. Add RGB values
point_cloud_with_rgb = add_rgb_to_pointcloud(point_cloud_sampled, rgb_image)
# Result: (1024, 6) array with [x, y, z, r, g, b]
```

### Step 3: Robot State Collection

```python
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R

# Initialize robot
robot = RobotInterface(ip_address="YOUR_ROBOT_IP")

# Get current pose
ee_pos, ee_quat = robot.get_ee_pose()
ee_pos = ee_pos.numpy()  # (3,) [x, y, z]
ee_euler = R.from_quat(ee_quat.numpy()).as_euler('XYZ')  # (3,) [roll, pitch, yaw]

# Get gripper state
gripper_state = get_gripper_state()  # -1.0 (open) to 1.0 (closed)

# Combine into state vector
state = np.concatenate([ee_pos, ee_euler, [gripper_state]])  # (7,)
```

### Step 4: Recording Loop

```python
import numpy as np
import zarr

# Initialize storage
point_cloud_list = []
state_list = []
action_list = []
episode_ends = []

episode_count = 0
frame_count = 0

while True:
    # 1. Get observations
    point_cloud, rgbd_frame = camera.get_frame()
    point_cloud = process_point_cloud(point_cloud)  # Transform, crop, downsample
    
    # 2. Get robot state
    state = get_robot_state()
    
    # 3. Human teleoperation (keyboard control)
    command = get_keyboard_command()
    
    if command == 'quit':
        episode_ends.append(frame_count)
        break
    
    if command == 'new_episode':
        episode_ends.append(frame_count)
        episode_count += 1
        continue
    
    # 4. Execute command and move robot
    delta_pose = command_to_delta(command)  # e.g., 'w' -> [0.03, 0, 0, 0, 0, 0]
    target_pose = state[:6] + delta_pose
    robot.move_to_ee_pose(target_pose[:3], euler_to_quat(target_pose[3:]))
    
    # 5. Get new state after movement
    new_state = get_robot_state()
    action = new_state - state  # Calculate delta
    
    # 6. Store data
    point_cloud_list.append(point_cloud)
    state_list.append(new_state)
    action_list.append(action)
    
    frame_count += 1
    
    print(f"Episode {episode_count}, Frame {frame_count}")

# Save to zarr
save_to_zarr(point_cloud_list, state_list, action_list, episode_ends, 
             save_path="data/datasets/source/your_task.zarr")
```

### Step 5: Normalization

```python
def normalize_data(state_arrays, action_arrays):
    """
    Normalize state and action arrays to expected ranges
    """
    # Compute workspace bounds from data
    pos_min = state_arrays[:, :3].min(axis=0)
    pos_max = state_arrays[:, :3].max(axis=0)
    
    # Normalize state to [-1, 1.5] (to match source data)
    state_normalized = state_arrays.copy()
    state_normalized[:, :3] = 2.5 * (state_arrays[:, :3] - pos_min) / (pos_max - pos_min) - 1.0
    
    # Normalize action to [-1, 1]
    action_normalized = action_arrays.copy()
    action_max = np.abs(action_arrays[:, :6]).max()
    action_normalized[:, :6] = action_arrays[:, :6] / action_max
    
    return state_normalized, action_normalized
```

### Step 6: Save to Zarr Format

```python
import zarr

def save_to_zarr(point_clouds, states, actions, episode_ends, save_path):
    """
    Save collected data to zarr format
    """
    # Create zarr root
    root = zarr.open(save_path, mode='w')
    
    # Create groups
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # Convert lists to arrays
    point_cloud_array = np.stack(point_clouds, axis=0)  # (T, 1024, 6)
    state_array = np.stack(states, axis=0)              # (T, 7)
    action_array = np.stack(actions, axis=0)            # (T, 7)
    
    # Normalize
    state_array, action_array = normalize_data(state_array, action_array)
    
    # Save to zarr
    data_group.create_dataset('point_cloud', data=point_cloud_array, 
                              chunks=(1, 1024, 6), dtype='float32')
    data_group.create_dataset('state', data=state_array, 
                              chunks=(100, 7), dtype='float32')
    data_group.create_dataset('action', data=action_array, 
                              chunks=(100, 7), dtype='float32')
    
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends), 
                              dtype='int64')
    
    print(f"✓ Saved {len(states)} frames across {len(episode_ends)} episodes to {save_path}")
```

---

## 🎮 Keyboard Control Reference

From `collect_demo.py`, the keyboard mapping:

### Position Control
- `w` - Move forward (+X)
- `s` - Move backward (-X)
- `a` - Move left (+Y)
- `d` - Move right (-Y)
- `e` - Move down (-Z)
- `r` - Move up (+Z)

### Orientation Control
- `z` - Rotate counterclockwise (-Yaw)
- `x` - Rotate clockwise (+Yaw)
- `c` - Pitch down (-Pitch)
- `v` - Pitch up (+Pitch)
- `b` - Roll left (-Roll)
- `n` - Roll right (+Roll)

### Gripper Control
- `y` - Open gripper
- `o` - Close gripper

### Recording Control
- `Space` - Hold position (no movement)
- `j` - Fine control mode (half speed)
- `q` - Save and quit
- `` ` `` - Quit without saving

**Movement amounts:**
- Normal mode: `delta_range = 0.03` meters, `delta_angle = 0.15` radians
- Fine mode (`j`): `delta_range = 0.015` meters

---

## ✅ Data Quality Checklist

Before using your collected data with DemoGen:

### 1. **Data Structure**
- [ ] Zarr file has `data/` and `meta/` groups
- [ ] `data/action` shape: `(T, 7)`
- [ ] `data/state` shape: `(T, 7)`
- [ ] `data/point_cloud` shape: `(T, 1024, 6)`
- [ ] `meta/episode_ends` contains correct frame indices

### 2. **Data Ranges**
- [ ] State values in `[-1.0, 1.5]` range
- [ ] Action values in `[-1.0, 1.0]` range
- [ ] Point cloud XYZ in workspace bounds (e.g., `[-0.3, 0.6]` meters)
- [ ] Point cloud RGB in `[0, 255]` range

### 3. **Episode Quality**
- [ ] Minimum 3 episodes recorded
- [ ] Each episode 80-120 frames (typical)
- [ ] Episodes have clear start/end states
- [ ] Smooth trajectories (no sudden jumps)
- [ ] Task completion visible in data

### 4. **Point Cloud Quality**
- [ ] Point clouds transformed to robot base frame
- [ ] Downsampled to exactly 1024 points
- [ ] Objects clearly visible in point cloud
- [ ] No NaN or inf values
- [ ] RGB colors properly aligned

### 5. **Action-State Consistency**
- [ ] `action[t]` ≈ `state[t+1] - state[t]`
- [ ] Action and state are NOT identical
- [ ] Gripper state changes at appropriate times

---

## 🔍 Validation Script

Use the inspection script to validate your collected data:

```bash
cd /home/hk/Documents/DemoGen
conda activate demogen
python inspect_source_data.py
```

Select your newly collected dataset and check:
1. ✓ Data shapes match expected format
2. ✓ Value ranges are correct
3. ✓ Action vs state difference analysis passes
4. ✓ Point clouds visualize correctly
5. ✓ Trajectories look smooth in 3D plot

---

## 🚀 Next Steps After Collection

Once you have validated source data:

1. **Create SAM masks:**
   ```bash
   cd data/sam_mask
   python get_mask.py --task your_task --text_prompt "your object description"
   ```

2. **Create config file:**
   ```bash
   cp demo_generation/demo_generation/config/flower.yaml \
      demo_generation/demo_generation/config/your_task.yaml
   ```
   
   Edit to match your task:
   - Update `source_name: your_task`
   - Update `mask_names` with your object descriptions
   - Adjust `trans_range` for your workspace
   - Set `parsing_frames` based on your trajectory

3. **Generate augmented demos:**
   ```bash
   cd demo_generation
   python gen_demo.py --config-name=your_task.yaml \
       generation.range_name=test \
       generation.n_gen_per_source=16 \
       generation.render_video=True
   ```

4. **Train policy:**
   ```bash
   cd ../diffusion_policies
   # Follow training instructions in docs/3_train_policies.md
   ```

---

## 📚 Reference Files

- **Data collection:** `real_world/collect_demo.py`
- **Environment setup:** `real_world/utils/panda_oyhand_env.py`
- **Camera interface:** `real_world/utils/realsense_camera.py`
- **Point cloud processing:** `real_world/utils/pcd_process.py`
- **Inspection tool:** `inspect_source_data.py`
- **Source data analysis:** `SOURCE_DATA_ANALYSIS.md`

---

## 🔧 Troubleshooting

### Point Cloud in Wrong Frame
**Problem:** Point clouds not aligned with robot movements

**Solution:** Check camera-to-robot transformation matrix. Use hand-eye calibration:
```python
# Example transformation
T_robot_camera = np.array([
    [R11, R12, R13, tx],
    [R21, R22, R23, ty],
    [R31, R32, R33, tz],
    [0,   0,   0,   1]
])
point_cloud_robot = (T_robot_camera @ point_cloud_camera.T).T
```

### Normalization Issues
**Problem:** Generated demos have wrong range

**Solution:** Match normalization to source data. Check workspace bounds:
```python
# Compute from your data
pos_min = states[:, :3].min(axis=0)
pos_max = states[:, :3].max(axis=0)
print(f"Workspace: X[{pos_min[0]:.3f}, {pos_max[0]:.3f}], "
      f"Y[{pos_min[1]:.3f}, {pos_max[1]:.3f}], "
      f"Z[{pos_min[2]:.3f}, {pos_max[2]:.3f}]")
```

### Jerky Trajectories
**Problem:** Robot movements not smooth

**Solution:** 
- Record at consistent frequency (10-30 Hz)
- Use fine control mode (`j` key) for precise movements
- Apply smoothing filter to trajectories
- Increase interpolation steps in DemoGen

### Point Cloud Too Dense/Sparse
**Problem:** Point cloud has wrong number of points

**Solution:** Use FPS (Farthest Point Sampling):
```python
from fpsample import fps_sampling
indices = fps_sampling(point_cloud, 1024)
point_cloud_sampled = point_cloud[indices]
```

---

## 💡 Best Practices

1. **Recording:**
   - Record multiple episodes (3-5 minimum)
   - Keep episodes similar in length (~100 frames)
   - Ensure good lighting for RGB-D
   - Start and end at consistent poses

2. **Quality:**
   - Check data immediately after recording
   - Visualize point clouds during collection
   - Verify robot reaches task goals
   - Remove failed episodes before saving

3. **Organization:**
   - Use descriptive task names
   - Document any special preprocessing
   - Save camera calibration parameters
   - Keep raw data backups

4. **Testing:**
   - Test with `generation.range_name=src` first (no transforms)
   - Gradually increase spatial augmentation range
   - Verify generated demos look reasonable
   - Check videos for trajectory quality

---

## 📞 Getting Help

If you encounter issues:
1. Check existing source data for reference: `data/datasets/source/flower.zarr`
2. Run inspection script: `python inspect_source_data.py`
3. Review collection code: `real_world/collect_demo.py`
4. Compare your data structure with working examples

Good luck with your data collection! 🎉
