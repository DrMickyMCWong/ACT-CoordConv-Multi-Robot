# Dataset Comparison Analysis

## Summary of Key Differences

### 1. File Naming Convention
- **New (raw_episodes)**: `raw_episode_0000_20251121_192511.h5` (includes timestamp)
- **Old (task7c)**: `episode_0.hdf5` (simple numbering, .hdf5 extension)

### 2. Data Structure
**New Format (raw_episodes):**
```
Root level keys: ['action', 'color_image', 'point_cloud', 'state', 'timestamps']
- action: (150, 6) float32
- color_image: (150, 720, 1280, 3) uint8
- point_cloud: (150, 4096, 6) float32
- state: (150, 6) float32
- timestamps: (150,) float64
```

**Old Format (task7c):**
```
Root level keys: ['action', 'observations']
- action: (400, 6) float32
- observations/
  - images/
    - front: (400, 480, 640, 3) uint8
  - qpos: (400, 6) float32
  - qvel: (400, 6) float32
```

### 3. Specific Differences

| Aspect | New (raw_episodes) | Old (task7c) | Notes |
|--------|-------------------|--------------|-------|
| Episode Length | 150 timesteps | 400 timesteps | New episodes are shorter |
| Image Resolution | 720x1280 | 480x640 | New images are higher res (2.25x larger) |
| Image Key | `color_image` | `observations/images/front` | Hierarchical structure in old |
| State Key | `state` | `observations/qpos` | Different naming |
| Velocity | Not present | `observations/qvel` | Missing in new format |
| Point Cloud | Present (150, 4096, 6) | Not present | Extra data in new format |
| Timestamps | Present (150,) | Not present | Extra metadata in new format |
| File Extension | `.h5` | `.hdf5` | Different but compatible |

### 4. Data Mapping Required

To convert new format → old format:
```
action → action (keep as is)
state → observations/qpos
color_image → observations/images/front (needs resizing 720x1280 → 480x640)
[generate zeros] → observations/qvel (6 values, all zeros like in task7c)
[discard] → point_cloud (not used in training)
[discard] → timestamps (not used in training)
```

### 5. Critical Conversion Steps

1. **Rename files**: `raw_episode_XXXX_timestamp.h5` → `episode_X.hdf5`
2. **Restructure HDF5 hierarchy**: Flat → Nested (observations group)
3. **Resize images**: 720x1280 → 480x640 (use cv2.resize or similar)
4. **Rename keys**: `state` → `qpos`, `color_image` → `images/front`
5. **Add qvel**: Create zeros array matching qpos shape
6. **Handle episode length**: 150 steps (shorter than task7c's 400)
7. **Remove unused data**: Drop point_cloud and timestamps

### 6. Questions to Resolve

- Should we resize images or update config to accept 720x1280?
- The new episodes are only 150 steps vs 400 - is this intentional?
- Should we pad episodes to 400 steps or adjust the training config?
