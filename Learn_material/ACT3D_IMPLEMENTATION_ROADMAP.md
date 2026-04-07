# ACT-3D Implementation Roadmap

**Multi-Modal Robot Learning with RGB + Point Clouds**

---

## 🎯 Goal

Extend ACT with 3D geometric understanding using PointNet encoder for point clouds, while keeping proven RGB pipeline.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ACT-3D Multi-Modal Policy                 │
└─────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
    RGB Branch                          Point Cloud Branch
          │                                     │
    ┌─────▼─────┐                         ┌────▼─────┐
    │  ResNet34 │                         │ PointNet │
    │  Encoder  │                         │ Encoder  │
    └─────┬─────┘                         └────┬─────┘
          │                                     │
    [512 features]                       [256 features]
          │                                     │
          └──────────────────┬──────────────────┘
                             │
                      Concatenate [768]
                             │
                    ┌────────▼────────┐
                    │   Transformer   │
                    │     Decoder     │
                    │  (ACT Standard) │
                    └────────┬────────┘
                             │
                      ┌──────▼──────┐
                      │ MLP Head    │
                      └──────┬──────┘
                             │
                    Action (6D joints)
```

---

## 📦 Implementation Checklist

### Phase 1: PointNet Encoder (Can do NOW, before camera)

- [ ] **Create `pointnet_encoder.py`**
  ```python
  class PointNetEncoder(nn.Module):
      """
      PointNet encoder for 3D point clouds
      Input: (B, N, 6) where N=1024, channels=[x,y,z,r,g,b]
      Output: (B, D) where D=256 global features
      """
      def __init__(self, in_channels=6, feature_dim=256):
          # Implement PointNet architecture:
          # 1. Shared MLP [6 -> 64 -> 128 -> 256]
          # 2. Max pooling across points
          # 3. Global feature vector
  ```

- [ ] **Test with dummy data**
  ```python
  # Generate random point clouds
  dummy_pc = torch.randn(8, 1024, 6)  # Batch=8, 1024 points
  encoder = PointNetEncoder()
  features = encoder(dummy_pc)
  assert features.shape == (8, 256)  # Verify output shape
  ```

- [ ] **Add normalization**
  - Center point clouds around workspace mean
  - Scale to unit sphere or workspace bounds
  - Separate normalization for XYZ and RGB

- [ ] **Optional: PointNet++ upgrade**
  - Hierarchical grouping
  - Better local feature extraction
  - Trade-off: Slower but more accurate

### Phase 2: Multi-Modal Policy Architecture

- [ ] **Modify `training/policy.py`**
  
  ```python
  class ACTPolicy3D(nn.Module):
      def __init__(self, config):
          super().__init__()
          
          # RGB encoder (existing)
          self.rgb_encoder = ResNet34(pretrained=True)
          self.rgb_features = 512
          
          # Point cloud encoder (NEW)
          self.pc_encoder = PointNetEncoder(
              in_channels=6,
              feature_dim=256
          )
          self.pc_features = 256
          
          # Concatenated features
          self.total_features = self.rgb_features + self.pc_features  # 768
          
          # Transformer decoder (existing, adjust input dim)
          self.transformer = TransformerDecoder(
              d_model=self.total_features,  # 768 instead of 512
              # ... other params
          )
          
          # MLP head (existing)
          self.action_head = MLPHead(...)
      
      def forward(self, rgb_obs, pc_obs, qpos, actions=None):
          """
          Args:
              rgb_obs: (B, T, 3, H, W) RGB images
              pc_obs: (B, T, 1024, 6) Point clouds
              qpos: (B, T, 6) Joint positions
              actions: (B, T, 6) Ground truth actions (training only)
          
          Returns:
              action_pred: (B, T, 6) Predicted actions
          """
          B, T = rgb_obs.shape[:2]
          
          # Encode RGB
          rgb_flat = rgb_obs.view(B*T, 3, H, W)
          rgb_features = self.rgb_encoder(rgb_flat)  # (B*T, 512)
          
          # Encode point clouds
          pc_flat = pc_obs.view(B*T, 1024, 6)
          pc_features = self.pc_encoder(pc_flat)  # (B*T, 256)
          
          # Concatenate features
          combined_features = torch.cat([rgb_features, pc_features], dim=-1)  # (B*T, 768)
          combined_features = combined_features.view(B, T, self.total_features)
          
          # Rest is same as original ACT
          # ... transformer decoder
          # ... action prediction
  ```

- [ ] **Create config flag**
  ```python
  # In config/config.py
  POLICY_CONFIG = {
      'use_point_cloud': True,  # Enable multi-modal
      'pointnet_features': 256,
      'rgb_features': 512,
      # ... existing config
  }
  ```

### Phase 3: Data Loading & Preprocessing

- [ ] **Update `training/utils.py`**
  
  ```python
  class MultiModalDataset(torch.utils.data.Dataset):
      def __init__(self, episode_ids, camera_names, norm_stats):
          # Load both RGB and point clouds
          
      def __getitem__(self, index):
          # Original data loading for RGB
          rgb_images = ...
          
          # NEW: Load point clouds
          point_clouds = dataset_root[f'/observations/point_cloud'][:]  # (T, 1024, 6)
          
          # Normalize point clouds
          point_clouds = self.normalize_point_cloud(point_clouds)
          
          return {
              'rgb': rgb_images,
              'point_cloud': point_clouds,
              'qpos': qpos,
              'actions': actions
          }
      
      def normalize_point_cloud(self, pc):
          """
          Normalize point cloud to robot workspace
          
          Args:
              pc: (T, 1024, 6) [x,y,z,r,g,b]
          
          Returns:
              pc_normalized: (T, 1024, 6) normalized
          """
          # Separate XYZ and RGB
          xyz = pc[..., :3]  # (T, 1024, 3)
          rgb = pc[..., 3:]  # (T, 1024, 3)
          
          # Normalize XYZ: center and scale
          xyz_mean = self.workspace_center  # e.g., [0.3, 0.0, 0.2]
          xyz_scale = self.workspace_scale  # e.g., 0.5 meters
          xyz_normalized = (xyz - xyz_mean) / xyz_scale
          
          # Normalize RGB: [0, 255] -> [0, 1]
          rgb_normalized = rgb / 255.0
          
          return np.concatenate([xyz_normalized, rgb_normalized], axis=-1)
  ```

- [ ] **Compute normalization statistics**
  ```python
  # Calculate workspace bounds from recorded data
  def compute_workspace_stats(dataset_dir):
      all_pc = []
      for episode in episodes:
          pc = load_point_clouds(episode)
          all_pc.append(pc[..., :3])  # XYZ only
      
      all_pc = np.concatenate(all_pc)
      workspace_center = np.mean(all_pc, axis=0)
      workspace_scale = np.std(all_pc)
      
      return workspace_center, workspace_scale
  ```

- [ ] **Add data augmentation**
  ```python
  def augment_point_cloud(pc, aug_config):
      """
      Augment point clouds for training robustness
      
      Args:
          pc: (1024, 6) [x,y,z,r,g,b]
      
      Returns:
          pc_augmented: (1024, 6)
      """
      xyz = pc[:, :3]
      rgb = pc[:, 3:]
      
      # Random rotation around Z-axis (robot base)
      if aug_config.get('rotate_z', False):
          angle = np.random.uniform(-np.pi/4, np.pi/4)
          xyz = rotate_z(xyz, angle)
      
      # Random jitter
      if aug_config.get('jitter', False):
          xyz += np.random.normal(0, 0.01, xyz.shape)
      
      # Random point dropout
      if aug_config.get('dropout', False):
          keep_mask = np.random.rand(1024) > 0.1
          xyz[~keep_mask] = 0
          rgb[~keep_mask] = 0
      
      return np.concatenate([xyz, rgb], axis=-1)
  ```

### Phase 4: Training Updates

- [ ] **Update training loop in `train.py`**
  ```python
  # Load both modalities
  batch = next(dataloader)
  rgb_data = batch['rgb'].to(device)
  pc_data = batch['point_cloud'].to(device)
  qpos = batch['qpos'].to(device)
  actions = batch['actions'].to(device)
  
  # Forward pass with both inputs
  action_pred = policy(
      rgb_obs=rgb_data,
      pc_obs=pc_data,  # NEW
      qpos=qpos,
      actions=actions
  )
  
  # Loss and backprop (same as before)
  loss = criterion(action_pred, actions)
  loss.backward()
  optimizer.step()
  ```

- [ ] **Add ablation modes**
  ```python
  # In config
  ABLATION_MODE = 'both'  # Options: 'rgb_only', 'pc_only', 'both'
  
  # In policy forward
  if config.ABLATION_MODE == 'rgb_only':
      features = rgb_features
  elif config.ABLATION_MODE == 'pc_only':
      features = pc_features
  elif config.ABLATION_MODE == 'both':
      features = torch.cat([rgb_features, pc_features], dim=-1)
  ```

### Phase 5: Camera Integration (When L515 arrives)

- [ ] **Create `camera_l515.py`**
  ```python
  import pyrealsense2 as rs
  import numpy as np
  
  class RealSenseL515:
      def __init__(self, width=640, height=480, fps=30):
          self.pipeline = rs.pipeline()
          config = rs.config()
          config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
          config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
          
          self.pipeline.start(config)
          
          # Get camera intrinsics
          profile = self.pipeline.get_active_profile()
          depth_profile = profile.get_stream(rs.stream.depth)
          self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
      
      def get_rgbd(self):
          """Get RGB and depth frames"""
          frames = self.pipeline.wait_for_frames()
          color_frame = frames.get_color_frame()
          depth_frame = frames.get_depth_frame()
          
          rgb = np.asanyarray(color_frame.get_data())  # (H, W, 3)
          depth = np.asanyarray(depth_frame.get_data())  # (H, W) in mm
          depth = depth / 1000.0  # Convert to meters
          
          return rgb, depth
      
      def get_point_cloud(self, depth_frame, color_frame):
          """Convert depth to point cloud"""
          pc_generator = rs.pointcloud()
          points = pc_generator.calculate(depth_frame)
          
          # Get vertices and colors
          vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
          colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3)
          
          # Combine
          point_cloud = np.concatenate([vertices, colors], axis=1)  # (N, 6)
          
          return point_cloud
  ```

- [ ] **Create `point_cloud_utils.py`**
  ```python
  import open3d as o3d
  import numpy as np
  
  def farthest_point_sampling(points, n_samples=1024):
      """
      Downsample point cloud using FPS
      
      Args:
          points: (N, 6) [x,y,z,r,g,b]
          n_samples: target number of points
      
      Returns:
          sampled_points: (n_samples, 6)
      """
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points[:, :3])
      
      # FPS
      sampled_indices = farthest_point_sample(pcd, n_samples)
      
      return points[sampled_indices]
  
  def transform_point_cloud(points, transformation_matrix):
      """
      Transform point cloud from camera to robot frame
      
      Args:
          points: (N, 6) [x,y,z,r,g,b] in camera frame
          transformation_matrix: (4, 4) camera-to-robot transform
      
      Returns:
          transformed_points: (N, 6) in robot frame
      """
      xyz = points[:, :3]
      rgb = points[:, 3:]
      
      # Apply transformation
      xyz_homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
      xyz_transformed = (transformation_matrix @ xyz_homogeneous.T).T[:, :3]
      
      return np.concatenate([xyz_transformed, rgb], axis=1)
  
  def filter_workspace(points, workspace_bounds):
      """
      Filter points outside robot workspace
      
      Args:
          points: (N, 6)
          workspace_bounds: dict with 'x', 'y', 'z' min/max
      
      Returns:
          filtered_points: (M, 6) where M <= N
      """
      xyz = points[:, :3]
      
      mask = (
          (xyz[:, 0] >= workspace_bounds['x'][0]) &
          (xyz[:, 0] <= workspace_bounds['x'][1]) &
          (xyz[:, 1] >= workspace_bounds['y'][0]) &
          (xyz[:, 1] <= workspace_bounds['y'][1]) &
          (xyz[:, 2] >= workspace_bounds['z'][0]) &
          (xyz[:, 2] <= workspace_bounds['z'][1])
      )
      
      return points[mask]
  ```

- [ ] **Hand-eye calibration**
  ```python
  # Collect calibration data: move robot to known poses,
  # record both robot position and camera view
  # Use cv2.calibrateHandEye() or similar
  
  def calibrate_camera_robot(calibration_data):
      """
      Compute transformation from camera to robot base
      
      Returns:
          T_robot_camera: (4, 4) transformation matrix
      """
      # Implement hand-eye calibration
      # Multiple approaches:
      # 1. Checkerboard in robot hand
      # 2. ArUco markers on robot base
      # 3. Manual correspondence
      
      return T_robot_camera
  ```

### Phase 6: Unified Recording

- [ ] **Create `record_episodes_unified.py`**
  ```python
  def record_episode(episode_idx, camera, follower, leader):
      """Record episode with RGB + point clouds"""
      
      rgb_images = []
      point_clouds = []
      joint_states = []
      actions = []
      
      for frame in range(num_frames):
          # Get camera data
          rgb, depth = camera.get_rgbd()
          pc_full = camera.depth_to_point_cloud(depth, rgb)
          
          # Filter and downsample
          pc_filtered = filter_workspace(pc_full, WORKSPACE_BOUNDS)
          pc_transformed = transform_point_cloud(pc_filtered, T_robot_camera)
          pc_sampled = farthest_point_sampling(pc_transformed, 1024)
          
          # Get robot data
          follower_state = get_robot_state(follower)
          leader_state = get_robot_state(leader)
          
          # Store
          rgb_images.append(rgb)
          point_clouds.append(pc_sampled)
          joint_states.append(follower_state['joint_state'])
          actions.append(leader_state['joint_state'])
      
      # Save to HDF5
      save_unified_hdf5(
          episode_idx,
          rgb_images,
          point_clouds,
          joint_states,
          actions
      )
  ```

---

## 🧪 Testing & Validation

### Unit Tests
- [ ] Test PointNet encoder with various input sizes
- [ ] Test point cloud normalization
- [ ] Test data loading pipeline
- [ ] Test multi-modal policy forward pass

### Integration Tests
- [ ] Record 5 test episodes
- [ ] Verify HDF5 structure
- [ ] Load and visualize point clouds
- [ ] Train for 10 epochs (sanity check)

### Ablation Studies
- [ ] RGB-only (baseline)
- [ ] PointCloud-only
- [ ] RGB + PointCloud (full model)
- [ ] Compare: PointNet vs PointNet++

---

## 📊 Expected Improvements

**Hypothesis**: Point clouds will help with:
1. ✅ **Grasp pose estimation**: Better 3D understanding of object orientation
2. ✅ **Occlusion handling**: 3D shape even when partially hidden
3. ✅ **Depth ambiguity**: Eliminate size/distance confusion
4. ✅ **Generalization**: 3D features more robust to viewpoint changes

**Trade-offs**:
- ⚖️ Slightly slower inference (PointNet forward pass)
- ⚖️ More parameters to train
- ⚖️ Requires camera calibration

**Target**: Match or exceed 90% success rate of ACT-RGB baseline

---

## 🛠️ Tools & Dependencies

```bash
# PyTorch Geometric (for point cloud operations)
pip install torch-geometric

# Open3D (for point cloud processing)
pip install open3d

# RealSense SDK (for L515)
pip install pyrealsense2

# Visualization
pip install matplotlib plotly
```

---

## 📚 References

- **PointNet**: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al., 2017)
- **3D Diffusion Policy**: Learning 3D diffusion policies with point clouds
- **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
- **PerAct**: Voxel-based 3D manipulation

---

**Ready to implement!** Start with Phase 1 (PointNet encoder) while waiting for camera. 🚀
