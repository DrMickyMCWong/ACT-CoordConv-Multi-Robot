#!/usr/bin/env python3
"""
Test training script for 3-camera ACT policy with real recorded data
"""
import sys
import os
sys.path.append('/home/hk/Documents/ACT_Shaka')
sys.path.append('/home/hk/Documents/ACT_Shaka/act-main')

import torch
import numpy as np
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import load_data, make_policy, make_optimizer

def test_training_with_real_data():
    """Test training pipeline with the recorded episode data"""
    print("=" * 60)
    print("Testing 3-Camera ACT Training Pipeline")
    print("=" * 60)
    
    # Check if episode data exists
    dataset_dir = '/home/hk/Documents/ACT_Shaka/act-main/act/dataset_dir/real_episodes'
    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False
    
    # Count episodes
    episode_files = [f for f in os.listdir(dataset_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    num_episodes = len(episode_files)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Found {num_episodes} episodes: {episode_files}")
    
    if num_episodes == 0:
        print("✗ No episode files found!")
        return False
    
    try:
        # Load data
        print(f"\nLoading data...")
        camera_names = POLICY_CONFIG['camera_names']
        batch_size_train = 2  # Small batch for testing
        batch_size_val = 2
        
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
        )
        
        print(f"✓ Data loaded successfully!")
        print(f"  Camera names: {camera_names}")
        print(f"  Is simulation: {is_sim}")
        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Val batches: {len(val_dataloader)}")
        print(f"  Norm stats keys: {list(norm_stats.keys())}")
        
        # Create policy and optimizer
        print(f"\nCreating policy and optimizer...")
        policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
        optimizer = make_optimizer(POLICY_CONFIG['policy_class'], policy)
        
        print(f"✓ Policy created: {POLICY_CONFIG['policy_class']}")
        print(f"✓ Optimizer created: {type(optimizer).__name__}")
        
        # Test one training batch
        print(f"\nTesting training step...")
        train_batch = next(iter(train_dataloader))
        image_data, qpos_data, action_data, is_pad = train_batch
        
        print(f"  Batch shapes:")
        print(f"    Images: {image_data.shape}")  # [batch, cameras, channels, height, width]
        print(f"    QPos: {qpos_data.shape}")     # [batch, state_dim]
        print(f"    Actions: {action_data.shape}") # [batch, episode_len, action_dim]
        print(f"    Padding: {is_pad.shape}")     # [batch, episode_len]
        
        # Move to GPU
        device = POLICY_CONFIG['device']
        image_data = image_data.to(device)
        qpos_data = qpos_data.to(device)
        action_data = action_data.to(device)
        is_pad = is_pad.to(device)
        
        # Forward pass
        policy.train()
        optimizer.zero_grad()
        
        loss_dict = policy(qpos_data, image_data, action_data, is_pad)
        loss = loss_dict['loss']
        
        print(f"  Loss computation:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value.item():.6f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed!")
        
        # Test validation batch
        print(f"\nTesting validation step...")
        val_batch = next(iter(val_dataloader))
        image_data, qpos_data, action_data, is_pad = val_batch
        
        # Move to GPU
        image_data = image_data.to(device)
        qpos_data = qpos_data.to(device)
        action_data = action_data.to(device)
        is_pad = is_pad.to(device)
        
        # Forward pass (no gradients)
        policy.eval()
        with torch.no_grad():
            loss_dict = policy(qpos_data, image_data, action_data, is_pad)
        
        print(f"  Validation loss computation:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value.item():.6f}")
        
        print(f"✓ Validation step completed!")
        
        # Test inference
        print(f"\nTesting inference...")
        with torch.no_grad():
            predicted_actions = policy(qpos_data, image_data)
        
        print(f"  Predicted actions shape: {predicted_actions.shape}")
        print(f"  Action range: [{predicted_actions.min().item():.3f}, {predicted_actions.max().item():.3f}]")
        
        print(f"✓ Inference completed!")
        
        print(f"\n" + "=" * 60)
        print(f"✓ ALL TRAINING TESTS PASSED!")
        print(f"✓ 3-camera ACT policy ready for full training")
        print(f"✓ Cameras: {camera_names}")
        print(f"✓ Episodes: {num_episodes}")
        print(f"=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during training test:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_with_real_data()
    sys.exit(0 if success else 1)