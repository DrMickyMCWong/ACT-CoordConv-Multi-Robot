# RoArm-M3 to DemoGen Data Format Conversion Guide

## Understanding Your Robot's Data Format

### Current RoArm-M3 Data (from `get_arm_data_immediate`):
```json
{
  "T": 1051,
  "x": 351.726223,        // mm
  "y": 8.634394135,       // mm  
  "z": 205.5705253,       // mm
  "tit": 0.056757289,     // End effector joint (rad)
  "b": 0.024543693,       // Base joint (rad)
  "s": 0.029145635,       // Shoulder joint (rad)
  "e": 1.589204096,       // Elbow joint (rad)
  "t": 0.009203885,       // Wrist joint 1 (rad)
  "r": -0.001533981,      // Wrist joint 2/Roll (rad)
  "g": 3.13392275,        // Gripper/End joint (rad)
  "tB": -72,              // Base load
  "tS": 92,               // Shoulder load
  "tE": 60,               // Elbow load
  "tT": 40,               // Wrist 1 load
  "tR": 20                // Wrist 2 load
}
```

### Required DemoGen Format:
```python
state = [x, y, z, euler_x, euler_y, euler_z, gripper]  # (7,) in meters and radians
action = state[t+1] - state[t]  # Delta movements
point_cloud = [...] # (1024, 6) RGB-D in robot frame
```

---

## Critical Questions & Analysis

### Q1: Can Joint Angles Calculate Euler Angles?

**Answer: PARTIALLY - Your robot uses 2-DOF orientation (roll + pitch)**

From `solver.hpp` analysis:

```cpp
// Forward Kinematics (line 305-322)
std::array<double, 5> computePosbyJointRad(
    double base_joint_rad,      // b
    double shoulder_joint_rad,  // s  
    double elbow_joint_rad,     // e
    double wrist_joint_rad,     // t
    double roll_joint_rad,      // r
    double hand_joint_rad       // g
) {
    // ... kinematic calculations ...
    lastT = (elbow_joint_rad + shoulder_joint_rad + wrist_joint_rad) - M_PI/2;
    // Returns: [x, y, z, roll, pitch]
    std::array<double, 5> result = {lastX, lastY, lastZ, roll_joint_rad, lastT};
    return result;
}
```

**Your robot's orientation is:**
- **Roll (r)**: Directly from `r` joint angle
- **Pitch (lastT)**: Calculated as `pitch = s + e + t - π/2`
- **Yaw**: NOT AVAILABLE (robot only has 2-DOF wrist)

### Q2: Mapping to DemoGen's 6-DOF Euler Angles

**Problem:** DemoGen expects `[roll, pitch, yaw]` but your robot only has 2-DOF.

**Solutions:**

#### Option A: Use 2-DOF with Yaw=0 (Recommended for simplicity)
```python
euler_x = r  # Roll from wrist joint 2
euler_y = s + e + t - math.pi/2  # Pitch from arm configuration  
euler_z = b  # Use base rotation as "yaw" (hacky but works)
```

#### Option B: Use Base as Yaw, Constrain Workspace (Better semantics)
```python
euler_x = r  # Roll
euler_y = s + e + t - math.pi/2  # Pitch
euler_z = 0.0  # Always 0 (2-DOF constraint)
# Handle base rotation separately or include in XY position
```

#### Option C: Convert to Rotation Matrix → Euler (Most accurate)
```python
# Use forward kinematics to get end effector rotation matrix
# Then convert to Euler angles
# More complex but handles arbitrary arm configurations
```

---

## Data Conversion Pipeline

### Step 1: Parse Robot Data

```python
import json
import numpy as np
import math

def parse_robot_data(json_data):
    """Parse RoArm-M3 JSON data"""
    data = json.loads(json_data)
    
    # Position (convert mm → meters)
    x = data['x'] / 1000.0  # mm → m
    y = data['y'] / 1000.0
    z = data['z'] / 1000.0
    
    # Joint angles (already in radians)
    b = data['b']   # Base
    s = data['s']   # Shoulder
    e = data['e']   # Elbow
    t = data['t']   # Wrist
    r = data['r']   # Roll
    g = data['g']   # Gripper/End effector
    
    return {
        'position': np.array([x, y, z]),
        'joints': {
            'base': b,
            'shoulder': s,
            'elbow': e,
            'wrist': t,
            'roll': r,
            'gripper': g
        }
    }
```

### Step 2: Calculate Orientation (Euler Angles)

```python
def compute_euler_from_joints(joints):
    """
    Compute Euler angles from joint configuration
    Based on solver.hpp forward kinematics
    """
    s = joints['shoulder']
    e = joints['elbow']
    t = joints['wrist']
    r = joints['roll']
    b = joints['base']
    
    # From solver.hpp line 320
    pitch = (e + s + t) - (math.pi / 2)
    roll = r
    
    # Option A: Use base as yaw (workspace constraint)
    yaw = b
    
    # Option B: Set yaw to zero (pure 2-DOF)
    # yaw = 0.0
    
    return np.array([roll, pitch, yaw])
```

### Step 3: Normalize Gripper State

```python
def normalize_gripper(gripper_rad, gripper_range=(0.0, 3.14)):
    """
    Normalize gripper from radians to [-1, 1]
    -1 = fully open, 1 = fully closed
    """
    g_min, g_max = gripper_range
    # Normalize to [0, 1]
    normalized = (gripper_rad - g_min) / (g_max - g_min)
    # Map to [-1, 1]
    return 2.0 * normalized - 1.0
```

### Step 4: Create State Vector

```python
def create_state_vector(robot_data):
    """
    Create DemoGen-compatible state vector
    Format: [x, y, z, euler_x, euler_y, euler_z, gripper]
    """
    parsed = parse_robot_data(robot_data)
    
    # Position (already in meters)
    position = parsed['position']
    
    # Orientation (Euler angles)
    euler = compute_euler_from_joints(parsed['joints'])
    
    # Gripper (normalized to [-1, 1])
    gripper = normalize_gripper(parsed['joints']['gripper'])
    
    # Combine into state vector
    state = np.concatenate([
        position,      # [x, y, z]
        euler,         # [roll, pitch, yaw]
        [gripper]      # gripper state
    ])
    
    return state  # Shape: (7,)
```

### Step 5: Compute Actions (Deltas)

```python
def compute_action(prev_state, current_state):
    """
    Compute action as delta between states
    """
    action = current_state - prev_state
    return action
```

---

## Updated Recording Script - Dual Format

### Modifications to `record_episodes_custom.py`:

```python
# Add at top of file
import json
import math
import zarr
import h5py
from pathlib import Path

def parse_robot_json(json_str):
    """
    Parse RoArm-M3 JSON data to extract BOTH joint and Cartesian states
    Returns: dict with both 'joint_state' and 'cartesian_state'
    """
    data = json.loads(json_str)
    
    # Position (mm → m)
    x = data['x'] / 1000.0
    y = data['y'] / 1000.0
    z = data['z'] / 1000.0
    
    # Joint angles (radians)
    b = data['b']   # Base
    s = data['s']   # Shoulder  
    e = data['e']   # Elbow
    t = data['t']   # Wrist
    r = data['r']   # Roll
    g = data['g']   # Gripper
    
    # Joint state vector (for ACT/HDF5)
    joint_state = np.array([b, s, e, t, r, g])
    
    # Compute Euler angles from joints (for DemoGen/Zarr)
    pitch = (e + s + t) - (math.pi / 2)
    roll = r
    yaw = b  # Using base as yaw
    
    # Normalize gripper to [-1, 1]
    gripper_normalized = 2.0 * (g / math.pi) - 1.0
    
    # Cartesian state vector [x, y, z, roll, pitch, yaw, gripper]
    cartesian_state = np.array([x, y, z, roll, pitch, yaw, gripper_normalized])
    
    return {
        'joint_state': joint_state,      # (6,) for ACT
        'cartesian_state': cartesian_state,  # (7,) for DemoGen
        'raw_position': np.array([x, y, z])  # (3,) for reference
    }

def get_robot_state_dual(robot):
    """
    Get robot state in BOTH formats
    Returns: dict with 'joint_state' and 'cartesian_state'
    """
    # Read raw serial data
    json_str = None
    for attempt in range(10):
        try:
            if robot.ser.in_waiting > 0:
                line = robot.ser.readline().decode('utf-8').strip()
                if line and line.startswith('{'):
                    test_parse = json.loads(line)
                    if 'x' in test_parse and 'b' in test_parse:
                        json_str = line
                        break
            time.sleep(0.01)
        except:
            continue
    
    if not json_str:
        return None
    
    # Parse to both formats
    return parse_robot_json(json_str)

# In your recording loop, KEEP and ADD:
# KEEP for HDF5 (ACT):
qpos_joint = follower_state['joint_state']  # (6,) joint angles
action_joint = leader_state['joint_state']  # (6,) joint angles

# ADD for Zarr (DemoGen):
state_cartesian = follower_state['cartesian_state']  # (7,) Cartesian pose
action_cartesian = leader_state['cartesian_state']  # (7,) Cartesian pose
```

---

## RGB-D Camera Integration - Dual Output

### Capture BOTH 4-channel images (ACT) AND point clouds (DemoGen):

```python
def capture_rgbd_dual_format(camera):
    """
    Capture RGB-D data in BOTH formats
    
    Returns:
        dict with:
        - 'rgbd_image': (H, W, 4) for ACT/HDF5 - RGB + Depth as 4th channel
        - 'point_cloud': (1024, 6) for DemoGen/Zarr - [x, y, z, r, g, b]
    """
    # 1. Get RGB-D from camera (e.g., Intel RealSense L515)
    rgb_image, depth_image = camera.get_rgbd()  # rgb: (H,W,3), depth: (H,W)
    
    # 2. Create 4-channel RGB-D image for ACT
    # Normalize depth to [0, 255] range or keep in meters
    depth_normalized = normalize_depth(depth_image)  # (H, W, 1)
    rgbd_image = np.concatenate([rgb_image, depth_normalized], axis=-1)  # (H, W, 4)
    
    # 3. Convert depth image to point cloud (camera frame)
    point_cloud_cam = depth_to_pointcloud(
        depth_image, 
        camera_intrinsics
    )
    
    # 4. Transform point cloud to robot base frame
    # You need camera-to-robot transformation matrix
    T_robot_camera = get_camera_calibration()  # 4x4 matrix
    point_cloud_robot = transform_points(point_cloud_cam, T_robot_camera)
    
    # 5. Add RGB colors to point cloud
    point_cloud_rgb = add_rgb_to_points(point_cloud_robot, rgb_image)
    
    # 6. Downsample to 1024 points for DemoGen
    point_cloud_1024 = downsample_fps(point_cloud_rgb, n_points=1024)
    
    return {
        'rgbd_image': rgbd_image,      # (H, W, 4) for ACT
        'point_cloud': point_cloud_1024  # (1024, 6) for DemoGen
    }

def normalize_depth(depth_image, max_depth=2.0):
    """
    Normalize depth for 4-channel image storage
    
    Args:
        depth_image: (H, W) depth in meters
        max_depth: maximum depth to consider (meters)
    
    Returns:
        (H, W, 1) normalized depth [0, 255]
    """
    # Clip and normalize
    depth_clipped = np.clip(depth_image, 0, max_depth)
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)
    return depth_normalized[..., np.newaxis]
```

---

## Complete Recording Loop - Dual Format

```python
def record_episode_dual_format(episode_idx, dataset_dir, num_frames=150):
    """
    Record one episode in BOTH ACT and DemoGen formats simultaneously
    
    Saves:
        - HDF5: data/episodes/episode_{idx}.hdf5 (for ACT)
        - Zarr: data/episodes_zarr/episode_{idx}.zarr (for DemoGen)
    """
    # Setup storage
    hdf5_path = Path(dataset_dir) / 'episodes' / f'episode_{episode_idx}.hdf5'
    zarr_path = Path(dataset_dir) / 'episodes_zarr' / f'episode_{episode_idx}.zarr'
    
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary storage for episode data
    joint_states = []       # For ACT/HDF5
    joint_actions = []      # For ACT/HDF5
    cartesian_states = []   # For DemoGen/Zarr
    cartesian_actions = []  # For DemoGen/Zarr
    rgbd_images = []        # For ACT/HDF5 (4-channel)
    point_clouds = []       # For DemoGen/Zarr
    
    prev_cartesian_state = None
    
    print(f"\n🎬 Recording Episode {episode_idx}")
    print(f"   Collecting {num_frames} frames...")
    
    for frame in range(num_frames):
        # 1. Get follower (current) state in BOTH formats
        follower_state = get_robot_state_dual(follower_robot)
        if not follower_state:
            print(f"⚠️ Failed to read follower state at frame {frame}")
            continue
        
        # 2. Get leader (action) state in BOTH formats
        leader_state = get_robot_state_dual(leader_robot)
        if not leader_state:
            print(f"⚠️ Failed to read leader state at frame {frame}")
            continue
        
        # 3. Capture camera data in BOTH formats
        camera_data = capture_rgbd_dual_format(camera)
        
        # 4. Store joint-space data (for ACT)
        joint_states.append(follower_state['joint_state'])      # (6,)
        joint_actions.append(leader_state['joint_state'])       # (6,)
        rgbd_images.append(camera_data['rgbd_image'])          # (H, W, 4)
        
        # 5. Store Cartesian-space data (for DemoGen)
        cartesian_states.append(follower_state['cartesian_state'])  # (7,)
        
        # Compute delta action for DemoGen
        if prev_cartesian_state is not None:
            delta_action = follower_state['cartesian_state'] - prev_cartesian_state
        else:
            delta_action = np.zeros(7)  # First frame
        cartesian_actions.append(delta_action)
        
        point_clouds.append(camera_data['point_cloud'])        # (1024, 6)
        
        prev_cartesian_state = follower_state['cartesian_state']
        
        # 6. Display progress
        if frame % 10 == 0:
            print(f"   Frame {frame}/{num_frames}")
        
        time.sleep(0.03)  # ~30Hz
    
    # Convert to numpy arrays
    joint_states = np.array(joint_states)          # (T, 6)
    joint_actions = np.array(joint_actions)        # (T, 6)
    cartesian_states = np.array(cartesian_states)  # (T, 7)
    cartesian_actions = np.array(cartesian_actions)  # (T, 7)
    rgbd_images = np.array(rgbd_images)            # (T, H, W, 4)
    point_clouds = np.array(point_clouds)          # (T, 1024, 6)
    
    # Save to HDF5 (ACT format with 4-channel images)
    print(f"💾 Saving HDF5 (ACT format)...")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('observations/qpos', data=joint_states)
        f.create_dataset('observations/images/top', data=rgbd_images)  # 4-channel
        f.create_dataset('action', data=joint_actions)
        f.attrs['sim'] = False
        f.attrs['compress'] = True
    
    # Save to Zarr (DemoGen format)
    print(f"💾 Saving Zarr (DemoGen format)...")
    root = zarr.open(zarr_path, mode='w')
    root.create_dataset('data/state', data=cartesian_states, 
                       chunks=(1, 7), dtype='float32')
    root.create_dataset('data/action', data=cartesian_actions,
                       chunks=(1, 7), dtype='float32')
    root.create_dataset('data/point_cloud', data=point_clouds,
                       chunks=(1, 1024, 6), dtype='float32')
    root.create_dataset('meta/episode_ends', data=np.array([len(cartesian_states)]))
    
    print(f"✅ Episode {episode_idx} saved successfully!")
    print(f"   HDF5: {hdf5_path}")
    print(f"   Zarr: {zarr_path}")
    print(f"   Frames: {len(joint_states)}")
    
    return {
        'episode_idx': episode_idx,
        'num_frames': len(joint_states),
        'hdf5_path': str(hdf5_path),
        'zarr_path': str(zarr_path)
    }
```

---

## Checklist Before Starting Dual-Format Recording

### Understanding Phase:
- [x] ✅ Confirm robot sends data in JSON format with x,y,z and joint angles
- [x] ✅ Verify joint angle units (radians confirmed)
- [x] ✅ Understand position coordinate system (robot base frame)
- [x] ✅ Verify Euler angle calculation: pitch = (e+s+t) - π/2
- [ ] Check gripper range (0 to π assumed, needs validation)

### Hardware Setup:
- [ ] RGB-D camera (Intel RealSense L515) mounted and connected
- [ ] Camera intrinsics parameters obtained
- [ ] Camera-to-robot transformation calibrated
- [x] ✅ Robot teleoperation working (leader-follower)
- [x] ✅ Torque disable/enable working for manual manipulation

### Software Modifications:
- [x] ✅ JSON parsing implemented and tested
- [x] ✅ Euler angle calculation verified with real robot
- [ ] RGB-D capture implementation (waiting for L515)
- [ ] Point cloud processing implementation
- [ ] Update recording script for dual-format storage
- [ ] Implement zarr file saving alongside HDF5

### Testing:
- [x] ✅ Test JSON parsing with sample data
- [x] ✅ Verify Euler angle calculation matches forward kinematics
- [ ] Test RGB-D capture and 4-channel image creation
- [ ] Test point cloud transformation to robot frame
- [ ] Record test episode and verify both formats
- [ ] Inspect HDF5 with existing tools
- [ ] Inspect Zarr with DemoGen's `inspect_source_data.py`

---

## Key Differences: ACT vs DemoGen vs Dual-Format

| Aspect | ACT (HDF5) | DemoGen (Zarr) | **Dual-Format (NEW)** |
|--------|------------|----------------|----------------------|
| **State Format** | Joint angles (6D) | Cartesian pose (7D: xyz, euler, gripper) | **BOTH stored** |
| **Action Format** | Joint angles | Delta pose | **BOTH stored** |
| **Observation** | RGB images | Point clouds (RGB-D) | **RGB-D (4-ch) + Point clouds** |
| **Storage** | HDF5 | Zarr | **HDF5 + Zarr simultaneously** |
| **Normalization** | Per-joint | Workspace-based | **Both methods preserved** |
| **Frame** | Joint space | Cartesian space | **Both frames stored** |
| **Image Format** | 3-channel RGB | Point cloud only | **4-channel RGB-D + Point cloud** |

### Dual-Format Storage Strategy

**Why store both?**
1. **ACT pipeline**: Continue using proven joint-space control with RGB-D images
2. **DemoGen pipeline**: Explore Cartesian-space control with point clouds
3. **Flexibility**: Compare performance between approaches
4. **Future-proof**: Can switch pipelines without re-recording data

**Storage overhead:**
- HDF5: ~50MB per episode (joint states + 4-channel images)
- Zarr: ~80MB per episode (Cartesian states + point clouds)
- Total: ~130MB per episode (acceptable for modern storage)

---

## Next Steps

1. **✅ COMPLETED: Euler Angle Calculation**
   - ✅ Implemented formulas from solver.hpp
   - ✅ Tested with real robot configurations
   - ✅ Verified pitch = (e+s+t) - π/2 calculation
   - ✅ Validated with `test_roarm_fk_real.py`

2. **⏳ WAITING: RGB-D Camera (Intel RealSense L515)**
   - Install camera drivers when hardware arrives
   - Calibrate camera intrinsics
   - Perform hand-eye calibration
   - Test RGB capture + point cloud generation

3. **📝 TODO: Implement PointNet Encoder for ACT-3D**
   - Create `pointnet_encoder.py` with PointNet architecture
   - Input: (B, 1024, 6) point clouds [x,y,z,r,g,b]
   - Output: (B, 256) 3D geometric features
   - Test with dummy data before camera arrives
   
4. **📝 TODO: Modify ACT Policy for Multi-Modal Input**
   - Update `training/policy.py`:
     - Keep ResNet34 for RGB → (B, 512) features
     - Add PointNet for point clouds → (B, 256) features
     - Concatenate → (B, 768) → Transformer decoder
   - Update `training/utils.py` for point cloud data loading
   - Add point cloud normalization and augmentation

5. **� TODO: Unified Recording Script**
   - Create `record_episodes_unified.py`
   - Store RGB images (H,W,3) + point clouds (1024,6) in single HDF5
   - Use validated `parse_robot_json()` for state extraction
   - Save: qpos, RGB images, point clouds, actions

6. **🧪 TODO: Test Multi-Modal Pipeline**
   - Record 10-20 test episodes
   - Train ACT-RGB (baseline, RGB only)
   - Train ACT-3D (RGB + PointNet features)
   - Compare success rates and behavior
   - Ablation: RGB-only vs PointCloud-only vs Both

7. **� TODO: Full Training & Evaluation**
   - Record 50-90 episodes
   - Train ACT-3D on full dataset
   - Evaluate on real robot
   - Compare to original ACT-RGB (90% baseline)
   - Document which tasks benefit from 3D understanding

8. **🎯 OPTIONAL: DemoGen Pipeline**
   - If ACT-3D works well, optionally try DemoGen
   - Use same point cloud data
   - Compare Cartesian vs Joint-space control

---

## Summary: Updated Architecture Strategy

**Key Decision**: Use **PointNet encoder** instead of simple depth channel!

### Why This Approach?
1. ✅ **More meaningful**: PointNet learns 3D geometry, not just pixel depths
2. ✅ **Efficient**: Point clouds (1024×6) much smaller than depth images (480×640)
3. ✅ **Modular**: Can keep proven ACT architecture, just add 3D branch
4. ✅ **Proven**: Similar to 3D Diffusion Policy, PerAct, RVT
5. ✅ **Flexible**: Can test RGB-only, PointCloud-only, or Both

### Architecture: ACT-3D Multi-Modal
```
RGB (H,W,3) ──→ ResNet34 ──→ [512 features] ──┐
                                                ├──→ Concat [768] ──→ Transformer ──→ Actions
Point Cloud (1024,6) ──→ PointNet ──→ [256 features] ──┘
```

### Data Format: Unified HDF5
```python
episode.hdf5:
  observations/qpos: (T, 6)           # Joint angles
  observations/images/top: (T,H,W,3)  # RGB only
  observations/point_cloud: (T,1024,6) # [x,y,z,r,g,b] in robot frame
  action: (T, 6)                       # Joint angles
```

**Current Status (Dec 26, 2025):**
- ✅ Euler angles validated
- ✅ Architecture designed (ACT-3D multi-modal)
- ⏳ Waiting for Intel RealSense L515
- 📝 Can implement PointNet encoder ahead of camera arrival

Would you like me to help implement the PointNet encoder or multi-modal policy architecture?

