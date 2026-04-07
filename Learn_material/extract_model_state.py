#!/usr/bin/env python3
"""
Script to extract just the model state_dict from the saved checkpoint
to avoid issues with missing classes like TrainingConfig.
"""

import torch
import pickle
import os

def extract_model_state():
    """Extract and save just the model state_dict."""
    
    # Path to the original model file
    original_path = "/home/hk/Documents/ACT_Shaka/reactive_pointnet_model.pth"
    new_path = "/home/hk/Documents/ACT_Shaka/reactive_pointnet_state_dict.pth"
    
    if not os.path.exists(original_path):
        print(f"❌ Original model file not found: {original_path}")
        return False
    
    try:
        # Create a custom unpickler that ignores missing classes
        class StateUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle missing classes by creating dummy classes
                if name == 'TrainingConfig':
                    class TrainingConfig:
                        pass
                    return TrainingConfig
                elif name == 'ReactivePointNet':
                    # We'll just ignore this and extract the state_dict
                    class DummyModel:
                        pass
                    return DummyModel
                return super().find_class(module, name)
        
        # Load with custom unpickler
        print("🔄 Loading original checkpoint...")
        with open(original_path, 'rb') as f:
            checkpoint = StateUnpickler(f).load()
        
        # Extract state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✓ Found model_state_dict from epoch {epoch}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✓ Found state_dict")
            else:
                # Maybe the checkpoint is the state_dict directly
                state_dict = checkpoint
                print(f"✓ Using checkpoint as state_dict")
        else:
            print(f"❌ Unexpected checkpoint format: {type(checkpoint)}")
            return False
        
        # Save just the state_dict
        print(f"💾 Saving clean state_dict to: {new_path}")
        torch.save(state_dict, new_path)
        
        print(f"✅ Successfully extracted model state_dict!")
        print(f"   - Original file: {original_path}")
        print(f"   - Clean file: {new_path}")
        print(f"   - State dict keys: {len(state_dict)} layers")
        
        # Print some info about the state dict
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   - Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting state_dict: {e}")
        return False

if __name__ == "__main__":
    success = extract_model_state()
    if success:
        print("\n✅ Model extraction complete!")
        print("Now you can use reactive_pointnet_state_dict.pth for loading")
    else:
        print("\n❌ Model extraction failed!")