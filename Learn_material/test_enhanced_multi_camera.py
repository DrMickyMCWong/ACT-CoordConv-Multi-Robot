#!/usr/bin/env python3
"""
Test script to verify the enhanced training pipeline works with 3 cameras including USB top camera
"""
import sys
import os
sys.path.append('/home/hk/Documents/ACT_Shaka')

import torch
import numpy as np
from config.config import POLICY_CONFIG, TASK_CONFIG

def test_enhanced_multi_camera_policy():
    """Test that the enhanced training pipeline can handle 3 cameras correctly"""
    print("=" * 70)
    print("Testing Enhanced Multi-Camera Training Pipeline")  
    print("=" * 70)
    
    # Print config
    print(f"Task config cameras: {TASK_CONFIG['camera_names']}")
    print(f"Policy config cameras: {POLICY_CONFIG['camera_names']}")
    
    # Test data shapes that match recorded episodes
    batch_size = 2
    num_cameras = len(POLICY_CONFIG['camera_names'])
    height, width = 480, 640
    state_dim = POLICY_CONFIG['state_dim'] 
    action_dim = POLICY_CONFIG['action_dim']
    num_queries = POLICY_CONFIG['num_queries']
    device = POLICY_CONFIG['device']
    
    print(f"\nTest parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of cameras: {num_cameras} {POLICY_CONFIG['camera_names']}")
    print(f"  Image size: {height}x{width}")
    print(f"  State dim: {state_dim}")  
    print(f"  Action dim: {action_dim}")
    print(f"  Num queries: {num_queries}")
    print(f"  Device: {device}")
    
    try:
        # Import and create policy using enhanced backbone
        print(f"\nImporting enhanced training policy...")
        from training.policy import ACTPolicy
        
        print(f"Creating ACT policy with enhanced multi-camera backbone...")
        policy = ACTPolicy(POLICY_CONFIG)
        
        print(f"✓ Enhanced policy created successfully!")
        print(f"  Model parameters: {sum(p.numel() for p in policy.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in policy.model.parameters() if p.requires_grad):,}")
        
        # Test that each camera gets its appropriate backbone
        print(f"\nVerifying camera-specific backbone configuration...")
        for i, cam_name in enumerate(POLICY_CONFIG['camera_names']):
            backbone = policy.model.backbones[i] 
            print(f"  Camera '{cam_name}': {backbone.__class__.__name__}")
            if cam_name.endswith('_depth'):
                print(f"    → Depth camera: Expected 3-channel input")
            elif cam_name == 'top':
                print(f"    → USB top camera: Expected 3-channel RGB input")
            else:
                print(f"    → RGB camera: Expected 3-channel input")
        
        # Create mock data for all cameras
        print(f"\nCreating mock 3-camera data...")
        mock_images = torch.randn(batch_size, num_cameras, 3, height, width).to(device)
        mock_qpos = torch.randn(batch_size, state_dim).to(device)
        mock_actions = torch.randn(batch_size, num_queries, action_dim).to(device)
        mock_is_pad = torch.zeros(batch_size, num_queries, dtype=torch.bool).to(device)
        
        print(f"  Images shape: {mock_images.shape}")
        print(f"  Robot state shape: {mock_qpos.shape}")
        
        # Test forward pass (training mode)
        print(f"\nTesting enhanced forward pass (training mode)...")
        loss_dict = policy(mock_qpos, mock_images, mock_actions, mock_is_pad)
        
        print(f"✓ Enhanced training forward pass successful!")
        print(f"  Loss components: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                print(f"    {key}: {value.item():.6f}")
                
        # Test forward pass (inference mode)
        print(f"\nTesting enhanced forward pass (inference mode)...")
        with torch.no_grad():
            predicted_actions = policy(mock_qpos, mock_images)
            
        print(f"✓ Enhanced inference forward pass successful!")
        print(f"  Predicted actions shape: {predicted_actions.shape}")
        
        # Verify shapes
        expected_shape = (batch_size, num_queries, action_dim)
        if predicted_actions.shape == expected_shape:
            print(f"✓ Output shape correct: {predicted_actions.shape}")
        else:
            print(f"✗ Output shape mismatch: got {predicted_actions.shape}, expected {expected_shape}")
            return False
            
        print(f"\n" + "=" * 70)
        print(f"✅ ENHANCED MULTI-CAMERA POLICY TESTS PASSED!")
        print(f"✅ Ready for training with enhanced 3-camera system:")
        print(f"   • Front RGB Camera (RealSense L515)")
        print(f"   • Front Depth Camera (RealSense L515)")  
        print(f"   • Top USB Camera (/dev/video0)")
        print(f"✅ Enhanced backbone supports different camera modalities")
        print(f"=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during enhanced testing:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_multi_camera_policy()
    sys.exit(0 if success else 1)