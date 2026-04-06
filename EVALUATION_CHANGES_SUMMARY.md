# Summary of Changes Made to evaluate_custom.py

## Changes Made:

### 1. ✅ Removed Windows-specific code
**Before:**
```python
import winsound
winsound.Beep(1000, 500)
```

**After:**
```python
# Removed winsound import
# Replaced beeps with print statements
```

### 2. ✅ Fixed image processing - REMOVED CROPPING
**Before (INCORRECT - had cropping):**
```python
x1, y1 = 90, 0
x2, y2 = 600, 480
image = image[y1:y2, x1:x2]  # CROP
image = cv2.resize(...)
```

**After (CORRECT - resize only):**
```python
# NO cropping - just resize to match training data
image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
```

### 3. ✅ Updated robot ports for Linux
**Before:**
```python
'leader': 'COM6',
'follower': 'COM5'
```

**After:**
```python
'leader': '/dev/ttyUSB1',
'follower': '/dev/ttyUSB0'
```

## Testing Checklist:

### Test with ORIGINAL task7c model first:
```bash
# In evaluate_custom.py, verify line 18:
# args.task default should be 'task7c'

# In config.py TRAIN_CONFIG:
'eval_ckpt_name': 'policy_best_epoch_4823_val_0.1500.ckpt'

# Run:
conda activate ACT
python evaluate_custom.py --task task7c
```

### Then test with NEW task7c_new model:
```bash
# In config.py TRAIN_CONFIG:
'eval_ckpt_name': 'policy_best_epoch_5303_val_0.0567.ckpt'

# Run:
python evaluate_custom.py --task task7c_new
```

## Important Configuration Values:

### For task7c (original):
- Episodes: 90, length: 400 timesteps
- Checkpoint: `policy_best_epoch_4823_val_0.1500.ckpt`
- Stats: `/checkpoints/task7c/dataset_stats.pkl`

### For task7c_new (newly trained):
- Episodes: 90, length: 150 timesteps
- Checkpoint: `policy_best_epoch_5303_val_0.0567.ckpt`
- Stats: `/checkpoints/task7c_new/dataset_stats.pkl`

## Potential Issues if Arm is Swinging Randomly:

1. **Wrong dataset_stats.pkl** - The normalization stats must match the training data
2. **Wrong checkpoint** - Make sure the checkpoint path is correct
3. **Camera resolution mismatch** - Camera must be capturing at expected resolution
4. **Initial position** - Robot should start from a reasonable position

## Quick Debug Test:
```python
# Check if stats are being loaded correctly
import pickle
stats_path = '/home/hk/Documents/ACT_Shaka/checkpoints/task7c/dataset_stats.pkl'
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)
print("qpos_mean:", stats['qpos_mean'])
print("qpos_std:", stats['qpos_std'])
print("action_mean:", stats['action_mean'])
print("action_std:", stats['action_std'])
```
