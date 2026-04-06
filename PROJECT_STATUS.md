# RoArm-M3 Dual-Pipeline Project Status

**Last Updated**: December 26, 2025

---

## 🎯 Project Goal

Train robot manipulation policies using **THREE approaches** for comparison:
1. **ACT-RGB Pipeline**: Joint-space control with RGB images (proven, 90% success)
2. **ACT-3D Pipeline**: Joint-space control with RGB + PointNet features (hybrid)
3. **DemoGen Pipeline**: Cartesian-space control with point clouds (experimental)

Store data in **unified format** with both RGB and point clouds to enable all approaches.

---

## 🔬 Technical Approaches

### Approach 1: ACT-RGB (Baseline)
**Current proven approach**
- **Input**: RGB images (480x640x3)
- **Encoder**: ResNet34 backbone
- **Control**: Joint-space (6D)
- **Status**: ✅ Working (90% success)

### Approach 2: ACT-3D (Hybrid - NEW RECOMMENDED)
**Combine vision and 3D geometry**
- **Input**: RGB images + Point clouds (1024x6)
- **Encoders**: 
  - ResNet34 for RGB → image features
  - **PointNet/PointNet++ for point clouds → 3D features**
- **Fusion**: Concatenate features before transformer
- **Control**: Joint-space (6D)
- **Advantages**:
  - Preserves proven ACT architecture
  - Adds 3D spatial understanding
  - Better grasp pose estimation
  - Occlusion handling via point clouds
  - Depth-aware manipulation

### Approach 3: DemoGen (Alternative)
**Full Cartesian-space approach**
- **Input**: Point clouds only (1024x6)
- **Encoder**: Point cloud transformer
- **Control**: Cartesian-space (7D: xyz + euler + gripper)
- **Advantages**:
  - Pure 3D reasoning
  - Workspace-based control
  - Data augmentation via DemoGen

---

## ✅ Completed Tasks

### Phase 1: ACT Training & Deployment (Nov-Dec 2024)
- ✅ Converted 90 episodes from Windows to Linux format
- ✅ Set up PyTorch 2.0.0 + CUDA 11.8 environment
- ✅ Trained ACT policy (10,000 epochs, best at epoch 5303)
- ✅ Deployed on Linux with 90% success rate
- ✅ Image processing pipeline validated (crop + resize)

### Phase 2: Euler Angle Validation (Dec 26, 2025)
- ✅ Created `test_roarm_fk_real.py` for FK testing
- ✅ Implemented torque disable/enable for manual manipulation
- ✅ Fixed serial data reading to get full JSON (x,y,z + joints)
- ✅ Validated Euler angle calculation: `pitch = (e+s+t) - π/2`
- ✅ Verified DemoGen state vector format (7D)
- ✅ Analyzed DemoGen source data format: point clouds (T, 1024, 6) [x,y,z,r,g,b]
- ✅ Decided on 3D approach: Use PointNet encoder instead of simple depth channel
- ✅ Updated architecture plan for ACT-3D hybrid model

**Test Results**:
```
Position: X=116.98mm, Y=-11.88mm, Z=-62.37mm
Joints: b=-0.1012, s=-0.0506, e=2.9130, t=-0.1549, r=-0.0752, g=3.1170 rad
Euler: roll=-4.31°, pitch=65.13°, yaw=-5.80°
DemoGen State: [0.1170, -0.0119, -0.0624, -0.0752, 1.1367, -0.1012, 0.9844]
```

---

## ⏳ Waiting For

### Hardware
- **Intel RealSense L515 Depth Camera** (ordered, awaiting arrival)
  - Will provide RGB-D for 4-channel images (ACT)
  - Will provide point clouds (DemoGen)

---

## 📝 Next Implementation Tasks

### 1. Camera Integration (when L515 arrives)
- [ ] Install Intel RealSense SDK 2.0
- [ ] Test RGB-D capture
- [ ] Verify depth quality and range
- [ ] Implement 4-channel image creation (RGB + normalized depth)
- [ ] Implement point cloud generation
- [ ] Calibrate camera intrinsics
- [ ] Perform hand-eye calibration (camera → robot base frame)

### 2. Unified Recording Script
- [ ] Create `record_episodes_unified.py`
- [ ] Implement `get_robot_state_dual()`:
  - Returns both joint state (6D) and Cartesian state (7D)
  - Based on validated `parse_robot_json()` from FK test
- [ ] Implement `capture_rgbd_unified()`:
  - Returns RGB image (H, W, 3)
  - Returns point cloud (1024, 6) [x,y,z,r,g,b] in robot frame
- [ ] Save in unified HDF5 format:
  - `observations/qpos` - (T, 6) joint angles
  - `observations/images/top` - (T, H, W, 3) RGB
  - `observations/point_cloud` - (T, 1024, 6) point cloud
  - `action` - (T, 6) joint angles
- [ ] Optionally save Zarr for DemoGen compatibility
- [ ] Test with dummy data before real recording

### 3. ACT-3D Architecture Implementation

**Core Modification: Multi-Modal Fusion**

- [ ] Create `pointnet_encoder.py`:
  - Implement PointNet or PointNet++ encoder
  - Input: (B, 1024, 6) point clouds
  - Output: (B, D) point cloud features (e.g., D=256)
  - Can use existing implementations (PyTorch Geometric, PointNet++)

- [ ] Modify `training/policy.py`:
  - Keep existing ResNet34 for RGB images → (B, 512) image features
  - Add PointNet encoder for point clouds → (B, 256) 3D features
  - Concatenate features: (B, 512+256) = (B, 768)
  - Feed concatenated features to transformer decoder
  
- [ ] Update data loading in `training/utils.py`:
  - Load both RGB images and point clouds
  - Normalize point clouds (center + scale)
  - Apply augmentations (point cloud jittering, rotation)

- [ ] Configuration in `config/config.py`:
  ```python
  POLICY_CONFIG = {
      'use_point_cloud': True,
      'pointnet_features': 256,
      'rgb_features': 512,
      # ... existing config
  }
  ```

**Architecture Diagram:**
```
RGB Image (H,W,3)  ───→  ResNet34  ───→  [512 features]
                                              ↓
                                         Concatenate → [768] → Transformer → Action
                                              ↑
Point Cloud (1024,6) ─→  PointNet  ───→  [256 features]
```

### 4. Evaluation & Comparison
- [ ] Train all three pipelines on same demonstrations:
  - **ACT-RGB** (baseline): RGB only
  - **ACT-3D** (hybrid): RGB + PointNet features
  - **DemoGen** (alternative): Point clouds only
- [ ] Test all policies on same tasks
- [ ] Compare metrics:
  - Success rate
  - Motion smoothness
  - Generalization to new positions/objects
  - Robustness to occlusions/lighting
  - Grasp pose accuracy
  - Inference speed (FPS)
- [ ] Ablation studies:
  - RGB only vs RGB+Point Cloud
  - PointNet vs PointNet++
  - Different feature fusion strategies
- [ ] Document findings
- [ ] Choose best approach or develop ensemble

---

## 📊 Data Format Specifications

### Unified HDF5 Format (For All Approaches)
```
episode_0.hdf5
├── observations/
│   ├── qpos          # (T, 6) joint angles [b,s,e,t,r,g]
│   ├── images/
│   │   └── top       # (T, H, W, 3) RGB images
│   └── point_cloud   # (T, 1024, 6) [x,y,z,r,g,b] in robot frame
├── action            # (T, 6) joint angles [b,s,e,t,r,g]
└── attrs/
    ├── sim: False
    └── compress: True
```

### Optional Zarr Format (DemoGen Compatibility)
```
episode_0.zarr
├── data/
│   ├── state         # (T, 7) [x,y,z, roll,pitch,yaw, gripper]
│   ├── action        # (T, 7) delta pose
│   └── point_cloud   # (T, 1024, 6) [x,y,z, r,g,b]
└── meta/
    └── episode_ends  # [T] cumulative episode lengths
```

### Point Cloud Format (CRITICAL)
- **Shape**: (1024, 6) per frame
- **Channels**: [x, y, z, r, g, b]
  - x, y, z: 3D coordinates in **robot base frame** (meters)
  - r, g, b: RGB color values (0-255 or normalized 0-1)
- **Sampling**: Farthest Point Sampling (FPS) from full depth image
- **Normalization**: Center around robot workspace, scale to unit sphere
- **Frame**: Must be transformed from camera frame to robot base frame using calibration

### Coordinate Conventions
- **Position**: meters (x,y,z in robot base frame)
- **Orientation**: radians (roll, pitch, yaw in XYZ Euler)
- **Gripper**: normalized [-1, 1] where -1=open, 1=closed
- **Euler angles**:
  - roll = r (wrist roll joint)
  - pitch = (e + s + t) - π/2 (computed from arm configuration)
  - yaw = b (base rotation)

---

## 🗂️ Key Files

### Working Scripts
- ✅ `test_roarm_fk_real.py` - FK test with torque control
- ✅ `robot.py` - Robot communication class
- ✅ `train.py` - ACT training script (needs update for point clouds)
- ✅ `evaluate_custom.py` - ACT evaluation/deployment

### To Be Created
- 📝 `record_episodes_unified.py` - Unified RGB + point cloud recording
- 📝 `camera_l515.py` - RealSense L515 interface
- 📝 `point_cloud_utils.py` - Point cloud processing (FPS, transformation)
- 📝 `pointnet_encoder.py` - PointNet/PointNet++ encoder
- 📝 `policy_multimodal.py` - ACT-3D multi-modal policy
- 📝 `calibration_camera_robot.py` - Hand-eye calibration

---

## 💾 Storage Requirements

### Per Episode (150 frames)
- **RGB Images**: ~30MB (150 frames × 480×640×3)
- **Point Clouds**: ~3.5MB (150 frames × 1024 points × 6 channels × 4 bytes)
- **Joint States**: ~0.01MB
- **Total per episode**: ~35MB (much smaller without redundant depth images!)

### For 90 Episodes
- **Total**: ~3.2GB (very manageable)

### Comparison to Original Plan
- ❌ Old plan (RGB-D images): ~11.7GB
- ✅ New plan (RGB + Point clouds): ~3.2GB
- **Savings**: 73% smaller! Point clouds are more efficient than full depth images.

---

## 🔍 Current Blockers

1. ⏳ **Hardware**: Waiting for Intel RealSense L515 camera
   - Cannot proceed with camera integration
   - Cannot test RGB-D capture
   - Cannot test point cloud generation
   - Cannot calibrate camera

2. ⏸️ **Software**: No blockers (validated Euler angles)
   - Can start implementing PointNet encoder ahead of time
   - Can prepare multi-modal policy architecture
   - Can study PointNet implementations

---

## 📈 Success Metrics

### Minimum Viable Product (MVP)
- [ ] Successfully record 10 episodes with RGB + point clouds
- [ ] HDF5 files validate with both modalities
- [ ] Train ACT-RGB baseline (existing approach)
- [ ] Train ACT-3D with PointNet features
- [ ] ACT-3D achieves >70% success rate

### Target Goals
- [ ] ACT-3D achieves >90% success rate (matching or exceeding ACT-RGB)
- [ ] Demonstrate improved grasp pose accuracy with point clouds
- [ ] Show robustness to lighting changes / occlusions
- [ ] Achieve real-time inference (>10 Hz)

### Stretch Goals
- [ ] ACT-3D outperforms ACT-RGB on complex tasks
- [ ] Identify which tasks benefit most from 3D understanding
- [ ] Test DemoGen pipeline and compare
- [ ] Develop hybrid approach if beneficial
- [ ] Publish findings or open-source framework

---

## 📅 Timeline Estimate

Assuming L515 arrives soon:

- **Week 1**: Camera setup, calibration, point cloud generation test
- **Week 2**: Unified recording implementation & PointNet encoder
- **Week 3**: Record 20-30 episodes with RGB + point clouds
- **Week 4**: Train ACT-RGB baseline + ACT-3D multi-modal
- **Week 5**: Evaluate and compare both approaches
- **Week 6**: Refinement, ablation studies, documentation

**Optional Week 7-8**: DemoGen pipeline if time permits

**Total**: ~6-8 weeks to comprehensive results

---

## 🎓 Key Learnings & Design Decisions

### Why PointNet instead of depth channel?
1. **More meaningful**: PointNet learns 3D geometric features, not just pixel depths
2. **Permutation invariant**: Handles unordered point clouds naturally
3. **Efficient**: 1024 points << 480×640 pixels (3.5MB vs 50MB per episode)
4. **Proven**: Used in robotics (3D Diffusion Policy, PerAct, etc.)
5. **Modular**: Can swap PointNet for PointNet++, transformers, etc.

### Why keep RGB?
1. **Texture/appearance**: Colors help identify objects
2. **Fine details**: High-resolution features for grasping
3. **Complementary**: Vision + geometry > either alone
4. **Proven baseline**: Keep what works (90% success)

### Why joint-space control?
1. **Direct motor control**: No IK errors
2. **Smooth trajectories**: Natural joint-space interpolation
3. **Proven**: ACT works well in joint space
4. **Can add Cartesian later**: Optional DemoGen comparison

---

## 📞 Questions to Resolve

1. ✅ Can joint angles calculate Euler angles? **YES - validated**
2. ⏳ Does L515 provide good point cloud quality? **Pending hardware**
3. 🤔 PointNet vs PointNet++? **Need to benchmark both**
4. 🤔 Best fusion strategy? **Concatenation vs cross-attention?**
5. ⏳ Will 3D features improve over RGB only? **Hypothesis: YES for grasping**
6. ⏳ Is hand-eye calibration difficult? **Will find out**
7. 🤔 How to normalize point clouds? **Center + scale to workspace bounds**

---

## 🔗 Related Work & References

### Point Cloud Encoders
- **PointNet**: Simple, fast, permutation-invariant (Qi et al., 2017)
- **PointNet++**: Hierarchical, better local features (Qi et al., 2017)
- **Point Transformer**: Attention-based, SOTA but slower
- **Recommendation**: Start with PointNet, upgrade to PointNet++ if needed

### Multi-Modal Robot Learning
- **3D Diffusion Policy**: Uses point clouds for manipulation
- **PerAct**: Voxelized point clouds + vision
- **ACT** (original): RGB images only
- **RVT**: Point cloud + rotation representations

### Implementations
- PyTorch Geometric: Point cloud layers
- PointNet PyTorch: Clean reference implementations
- Open3D: Point cloud processing utilities

---

**Status**: Ready to implement ACT-3D architecture! Waiting for L515 camera. 🚀
