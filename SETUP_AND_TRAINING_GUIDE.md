# ACT Environment Setup and Training Guide

## Current Status

### ✅ Completed
1. Dataset conversion: 90 episodes converted to ACT format
2. Converted dataset location: `/home/hk/Documents/ACT_Shaka/data/task7c_new/`
3. GPU verification script created: `verify_gpu.py`

### 🔄 In Progress
- Conda environment creation (this takes 5-15 minutes)

### ⏳ To Do
1. Verify GPU functionality
2. Update config files for Linux paths
3. Run test training

---

## Quick Start Commands (After Conda Environment is Ready)

### 1. Activate Environment
```bash
conda activate ACT
```

### 2. Verify GPU
```bash
python verify_gpu.py
```

Expected output: Should show CUDA available, GPU name, and pass all tests.

### 3. Update Config for New Dataset
Edit `config/config.py` and change:
```python
# OLD (Windows paths)
DATA_DIR = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data'
CHECKPOINT_DIR = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\checkpoints'

# NEW (Linux paths)
DATA_DIR = '/home/hk/Documents/ACT_Shaka/data'
CHECKPOINT_DIR = '/home/hk/Documents/ACT_Shaka/checkpoints'
```

### 4. Run Training on New Dataset
```bash
python train.py --task task7c_new
```

This will:
- Load 90 converted episodes from `data/task7c_new/`
- Split into 72 training + 18 validation episodes
- Train ACT policy with GPU acceleration
- Save checkpoints to `checkpoints/task7c_new/`

---

## Training Configuration

From `config/config.py`:
- **Episode length**: 1200 (but your episodes are 150, which is fine)
- **State/Action dims**: 6 (matches your data ✓)
- **Camera**: Single front camera at 480x640 (matches converted data ✓)
- **Batch size**: 32 train, 32 validation
- **Max epochs**: 10,000 (with early stopping based on validation loss)
- **Learning rate**: 5e-5

---

## Expected Training Time

With GPU acceleration:
- **Per epoch**: ~30 seconds - 2 minutes (depends on GPU)
- **To convergence**: Typically 500-2000 epochs
- **Total time**: 4-12 hours (estimate)

Without GPU (CPU only):
- **Per epoch**: 10-30 minutes
- **Total time**: Days (not recommended)

---

## Monitoring Training

Training will print:
```
Epoch 0
Val loss:   0.15234
loss: 0.156 kl: 0.123 ...
Train loss: 0.15678
loss: 0.158 kl: 0.125 ...
```

Good signs:
- ✓ Loss decreasing over time
- ✓ Validation loss tracking training loss
- ✓ Learning rate adjustments when loss plateaus

Bad signs:
- ✗ Loss = NaN (training failed, reduce learning rate)
- ✗ Validation >> Training loss (overfitting)
- ✗ Loss not decreasing after 100+ epochs

---

## Saved Outputs

Training creates:
1. **Checkpoints**: `checkpoints/task7c_new/policy_epoch_*.ckpt`
2. **Best model**: `checkpoints/task7c_new/policy_best_epoch_*_val_*.ckpt`
3. **Dataset stats**: `checkpoints/task7c_new/dataset_stats.pkl` (for deployment)
4. **Training curves**: `checkpoints/task7c_new/train_val_*_seed_42.png`

---

## Next Steps After Training

1. Evaluate trained policy on test episodes
2. Deploy to robot (remember to resize images to 480x640!)
3. Fine-tune if performance is not satisfactory

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in config:
```python
'batch_size_train': 16,  # was 32
'batch_size_val': 16,    # was 32
```

### "No module named 'training'"
Make sure you're in the project root:
```bash
cd /home/hk/Documents/ACT_Shaka
```

### Training is very slow
Check GPU is being used:
```bash
nvidia-smi  # Should show python process using GPU
```

### Loss stays high
- May need more data (90 episodes might be borderline)
- Check data quality (consistent demonstrations?)
- Try adjusting learning rate or model size
