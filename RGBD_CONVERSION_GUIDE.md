# Guide: Converting ACT to Support RGBD (4-Channel) Images

## Summary of Required Changes

To support RGBD (RGB + Depth) images, you need to modify **4 key areas**:

1. **Backbone** - Modify ResNet's first conv layer to accept 4 channels
2. **Config** - Update image channel configuration
3. **Data Loading** - Handle 4-channel images in dataset
4. **Evaluation** - Capture and process depth channel

---

## 1. Modify Backbone (detr/models/backbone.py)

### Current Issue:
ResNet expects 3-channel RGB input. The first conv layer has:
```python
Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
```

### Solution: Modify the first convolution layer

Add this method to the `Backbone` class (after line 96):

```python
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 in_channels: int = 3):  # ADD THIS PARAMETER
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        
        # MODIFY FIRST CONV LAYER FOR RGBD
        if in_channels != 3:
            # Save original weights
            original_conv1 = backbone.conv1
            # Create new conv1 with desired input channels
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new conv1 weights
            with torch.no_grad():
                if in_channels == 4:  # RGBD case
                    # Copy RGB weights
                    backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                    # Initialize depth channel (average of RGB or zeros)
                    backbone.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
                    # Alternative: Use zeros for depth channel
                    # backbone.conv1.weight[:, 3:4, :, :] = 0
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
```

### Update build_backbone function (around line 115):

```python
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    # ADD in_channels parameter
    in_channels = getattr(args, 'in_channels', 3)  # Default to 3 for backward compatibility
    
    backbone = Backbone(
        args.backbone, 
        train_backbone, 
        return_interm_layers, 
        args.dilation,
        in_channels=in_channels  # PASS THE PARAMETER
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
```

---

## 2. Update Config (config/config.py)

Add image channel configuration to POLICY_CONFIG:

```python
# policy config
POLICY_CONFIG = {
    'lr': 5e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 100,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 5e-5,
    'backbone': 'resnet34',
    'enc_layers': 5,
    'dec_layers': 8,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': False,
    'in_channels': 4,  # ADD THIS: 3 for RGB, 4 for RGBD
}
```

Also update TASK_CONFIG if needed:

```python
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 1200,
    'state_dim': 6,
    'action_dim': 6,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0,
    'image_channels': 4,  # ADD THIS: Number of image channels
}
```

---

## 3. Update Policy Loading (training/policy.py)

Find where the model is created and pass in_channels:

```python
# In ACTPolicy.__init__ or similar
def __init__(self, args_override):
    # ... existing code ...
    
    # Create backbone with correct input channels
    backbone_args = argparse.Namespace()
    backbone_args.backbone = self.backbone
    backbone_args.lr_backbone = self.lr_backbone
    backbone_args.masks = False
    backbone_args.dilation = False
    backbone_args.position_embedding = 'sine'
    backbone_args.hidden_dim = self.hidden_dim
    backbone_args.in_channels = args_override.get('in_channels', 3)  # ADD THIS
    
    self.model = build_detr(backbone_args)
```

---

## 4. Update Data Handling

### 4.1 Recording Scripts (record_episodes*.py)

Modify capture_image function to capture RGBD:

```python
def capture_rgbd_image(cam):
    """Capture RGB and Depth from depth camera"""
    # For RealSense or similar depth camera
    ret, frame = cam.read()
    
    # If using pyrealsense2:
    # import pyrealsense2 as rs
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # depth_frame = frames.get_depth_frame()
    # color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get depth (depends on your camera API)
    # For demonstration, assuming depth is available:
    # depth = get_depth_from_camera()  # Your camera-specific code
    
    # Normalize depth to 0-255 range
    # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Stack RGB + D to create 4-channel image
    # rgbd = np.concatenate([rgb, depth_normalized[:, :, np.newaxis]], axis=2)
    
    # Crop and resize (same as before)
    # x1, y1, x2, y2 = 90, 0, 600, 480
    # rgbd = rgbd[y1:y2, x1:x2]
    # rgbd = cv2.resize(rgbd, (640, 480))
    
    return rgbd  # Shape: (480, 640, 4)
```

### 4.2 Data Storage (HDF5)

Images should be stored as (H, W, 4) instead of (H, W, 3):

```python
# In recording script
with h5py.File(filepath, 'w') as root:
    # ...
    for cam_name in camera_names:
        _ = image.create_dataset(
            cam_name, 
            (max_timesteps, cam_height, cam_width, 4),  # CHANGED: 4 channels
            dtype='uint8',
            chunks=(1, cam_height, cam_width, 4)
        )
```

### 4.3 Evaluation (evaluate_custom.py)

Update capture_image to return RGBD:

```python
def capture_image(cam):
    """Capture RGBD image"""
    _, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get depth from your camera
    # depth = get_depth()
    # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Combine
    # rgbd = np.concatenate([rgb, depth_normalized[:, :, np.newaxis]], axis=2)
    
    # Crop
    x1, y1, x2, y2 = 90, 0, 600, 480
    rgbd = rgbd[y1:y2, x1:x2]
    
    # Resize
    rgbd = cv2.resize(rgbd, (640, 480), interpolation=cv2.INTER_AREA)
    
    return rgbd  # Shape: (480, 640, 4)
```

---

## 5. Data Normalization Consideration

The normalization in training/utils.py divides by 255.0:

```python
image_data = image_data / 255.0
```

This works for all channels (RGB + normalized depth). If your depth values are already in a different range, you may need separate normalization:

```python
# Split channels
rgb_data = image_data[:, :3, :, :]  # First 3 channels
depth_data = image_data[:, 3:4, :, :]  # 4th channel

# Normalize separately if needed
rgb_data = rgb_data / 255.0
depth_data = depth_data / depth_max  # Or keep /255.0 if already normalized

# Recombine
image_data = torch.cat([rgb_data, depth_data], dim=1)
```

---

## Implementation Checklist

- [ ] 1. Modify `detr/models/backbone.py` - Add in_channels parameter
- [ ] 2. Update `config/config.py` - Add 'in_channels': 4 to POLICY_CONFIG
- [ ] 3. Update `training/policy.py` - Pass in_channels when building model
- [ ] 4. Update recording scripts - Capture RGBD (4 channels)
- [ ] 5. Update HDF5 storage - Change shape from (H,W,3) to (H,W,4)
- [ ] 6. Update evaluation - Capture RGBD during inference
- [ ] 7. Test data loading - Verify 4-channel images load correctly
- [ ] 8. Retrain model - Train from scratch with RGBD data

---

## Testing Steps

### 1. Test backbone modification:
```python
import torch
from detr.models.backbone import Backbone

backbone = Backbone('resnet34', train_backbone=True, 
                   return_interm_layers=False, dilation=False, 
                   in_channels=4)

# Test with 4-channel input
test_input = torch.randn(1, 4, 480, 640)  # Batch=1, Channels=4, H=480, W=640
output = backbone(test_input)
print(f"Output shape: {output['0'].shape}")  # Should work without errors
```

### 2. Test data loading:
```python
# Record a test episode with RGBD
# Load it back and check shape
import h5py
with h5py.File('test_episode.hdf5', 'r') as f:
    img = f['observations/images/front'][0]
    print(f"Image shape: {img.shape}")  # Should be (480, 640, 4)
```

### 3. Test end-to-end:
- Record few RGBD episodes
- Train for 10 epochs
- Verify no shape mismatch errors

---

## Notes

1. **Pretrained weights**: The RGB channels will use ImageNet pretrained weights, but the depth channel will be randomly initialized (or initialized as average of RGB).

2. **Training time**: May need more epochs for the depth channel to learn useful features.

3. **Alternative**: Instead of modifying conv1, you could add a separate 1x1 conv to project 4→3 channels before the backbone.

4. **Backward compatibility**: The `in_channels` parameter defaults to 3, so existing RGB models still work.
