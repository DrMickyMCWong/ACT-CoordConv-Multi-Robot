#!/usr/bin/env python3
"""
Test the enhanced training pipeline with real recorded 3-camera data
"""
import sys
import os
sys.path.append('/home/hk/Documents/ACT_Shaka')

import torch
from config.config import POLICY_CONFIG, TASK_CONFIG
from training.utils import load_data, make_policy

def test_real_data_training():
    """Test training pipeline with real recorded 3-camera episodes"""
    print("=" * 70)
    print("Testing Enhanced Training Pipeline with Real 3-Camera Data")
    print("=" * 70)
    
    # Configuration
    dataset_dir = "/home/hk/Documents/ACT_Shaka/act-main/act/dataset_dir/real_episodes"
    num_episodes = 12
    camera_names = POLICY_CONFIG['camera_names']  # ['front', 'front_depth', 'top']
    batch_size_train = 2
    batch_size_val = 1
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Expected episodes: {num_episodes}")
    print(f"Camera names: {camera_names}")
    
    try:
        # Load data
        print(f"\nLoading 3-camera episode data...")
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
        )
        
        print(f"✓ Data loaded successfully!")
        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Val batches: {len(val_dataloader)}")
        print(f"  Is simulation: {is_sim}")
        print(f"  Norm stats keys: {list(norm_stats.keys())}")
        
        # Create enhanced policy
        print(f"\nCreating enhanced policy...")
        policy = make_policy('ACT', POLICY_CONFIG)
        print(f"✓ Enhanced policy created!")
        
        # Test with real batch
        print(f"\nTesting with real 3-camera batch...")
        for batch_idx, (image_data, qpos_data, action_data, is_pad) in enumerate(train_dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Real image data: {image_data.shape}")
            print(f"  Real qpos data: {qpos_data.shape}")
            print(f"  Real action data: {action_data.shape}")
            print(f"  Real padding: {is_pad.shape}")
            
            # Move to GPU
            image_data = image_data.cuda()
            qpos_data = qpos_data.cuda()
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            
            # Forward pass
            loss_dict = policy(qpos_data, image_data, action_data, is_pad)
            
            print(f"  Training loss:")
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    print(f"    {key}: {value.item():.6f}")
            
            # Test inference
            with torch.no_grad():
                predicted_actions = policy(qpos_data, image_data)
                print(f"  Inference output: {predicted_actions.shape}")
            
            # Only test first batch for speed
            break
            
        # Test validation batch
        print(f"\nTesting with real validation batch...")
        for batch_idx, (image_data, qpos_data, action_data, is_pad) in enumerate(val_dataloader):
            image_data = image_data.cuda()
            qpos_data = qpos_data.cuda() 
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            
            with torch.no_grad():
                loss_dict = policy(qpos_data, image_data, action_data, is_pad)
                print(f"  Validation loss:")
                for key, value in loss_dict.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: {value.item():.6f}")
            break
            
        print(f"\n" + "=" * 70)
        print(f"✅ REAL 3-CAMERA DATA TESTS PASSED!")
        print(f"✅ Enhanced training pipeline ready for:")
        print(f"   • 12 recorded episodes")
        print(f"   • 3-camera setup (front RGB + depth + USB top)")
        print(f"   • Full training with real robot data")
        print(f"=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during real data testing:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_data_training()
    sys.exit(0 if success else 1)