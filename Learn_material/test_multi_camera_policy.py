#!/usr/bin/env python3
"""
Test script to verify 3-camera ACT policy setup works correctly
"""
import sys
import os
sys.path.append('/home/hk/Documents/ACT_Shaka')
sys.path.append('/home/hk/Documents/ACT_Shaka/act-main')

import torch
import numpy as np
from config.config import POLICY_CONFIG, TASK_CONFIG

def test_multi_camera_policy():
    """Test that ACT policy can handle 3 cameras correctly"""
    print("=" * 60)
    print("Testing Multi-Camera ACT Policy Setup")
    print("=" * 60)
    
    # Print config
    print(f"Task config cameras: {TASK_CONFIG['camera_names']}")
    print(f"Policy config cameras: {POLICY_CONFIG['camera_names']}")
    
    # Test data shapes that match what's recorded
    batch_size = 2
    num_cameras = len(POLICY_CONFIG['camera_names'])
    height, width = 480, 640
    state_dim = POLICY_CONFIG['state_dim']
    action_dim = POLICY_CONFIG['action_dim']
    num_queries = POLICY_CONFIG['num_queries']
    
    print(f"\nTest parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of cameras: {num_cameras}")
    print(f"  Image size: {height}x{width}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Num queries: {num_queries}")
    
    # Create mock input data
    print(f"\nCreating mock data...")
    
    device = POLICY_CONFIG['device']
    print(f"  Using device: {device}")
    
    # Mock image data: [batch, num_cameras, channels, height, width]
    mock_images = torch.randn(batch_size, num_cameras, 3, height, width).to(device)
    print(f"  Images shape: {mock_images.shape}")
    
    # Mock robot state: [batch, state_dim]  
    mock_qpos = torch.randn(batch_size, state_dim).to(device)
    print(f"  Robot state shape: {mock_qpos.shape}")
    
    # Mock actions for training: [batch, num_queries, action_dim]
    mock_actions = torch.randn(batch_size, num_queries, action_dim).to(device)
    print(f"  Actions shape: {mock_actions.shape}")
    
    # Mock padding mask: [batch, num_queries]
    mock_is_pad = torch.zeros(batch_size, num_queries, dtype=torch.bool).to(device)
    print(f"  Padding mask shape: {mock_is_pad.shape}")
    
    try:
        # Import and create policy
        print(f"\nImporting ACT policy...")
        from training.policy import ACTPolicy
        
        print(f"Creating ACT policy with config...")
        policy = ACTPolicy(POLICY_CONFIG)
        
        print(f"✓ Policy created successfully!")
        print(f"  Model parameters: {sum(p.numel() for p in policy.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in policy.model.parameters() if p.requires_grad):,}")
        
        # Test forward pass (training mode)
        print(f"\nTesting forward pass (training mode)...")
        loss_dict = policy(mock_qpos, mock_images, mock_actions, mock_is_pad)
        
        print(f"✓ Training forward pass successful!")
        print(f"  Loss components: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                print(f"    {key}: {value.item():.6f}")
            else:
                print(f"    {key}: {value}")
        
        # Test forward pass (inference mode)
        print(f"\nTesting forward pass (inference mode)...")
        with torch.no_grad():
            predicted_actions = policy(mock_qpos, mock_images)
        
        print(f"✓ Inference forward pass successful!")
        print(f"  Predicted actions shape: {predicted_actions.shape}")
        print(f"  Expected shape: [{batch_size}, {num_queries}, {action_dim}]")
        
        # Verify shapes match expectations
        expected_action_shape = (batch_size, num_queries, action_dim)
        if predicted_actions.shape == expected_action_shape:
            print(f"✓ Output shape matches expected!")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"  Got: {predicted_actions.shape}")
            print(f"  Expected: {expected_action_shape}")
            return False
            
        print(f"\n" + "=" * 60)
        print(f"✓ ALL TESTS PASSED! Multi-camera ACT policy setup works correctly.")
        print(f"✓ Ready for training with 3 cameras: {POLICY_CONFIG['camera_names']}")
        print(f"=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_camera_policy()
    sys.exit(0 if success else 1)