#!/usr/bin/env python3
"""
Verify GPU and CUDA functionality with PyTorch.
Run this script inside the ACT conda environment.
"""

import sys
import torch
import torchvision

print("="*60)
print("PyTorch GPU Verification")
print("="*60)

# Python version
print(f"\nPython version: {sys.version}")

# PyTorch version
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Total Memory: {total_mem:.2f} GB")
        
    # Set default device
    print(f"\nCurrent device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test tensor operations on GPU
    print("\n" + "="*60)
    print("Testing GPU Operations")
    print("="*60)
    
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        print("✓ Created tensor on GPU")
        
        # Perform operation
        y = x @ x.T
        print("✓ Matrix multiplication on GPU successful")
        
        # Move to CPU
        y_cpu = y.cpu()
        print("✓ Moved tensor from GPU to CPU")
        
        # Test with ResNet (common in ACT)
        print("\n" + "="*60)
        print("Testing ResNet on GPU")
        print("="*60)
        
        from torchvision.models import resnet18
        model = resnet18(pretrained=False).cuda()
        print("✓ Loaded ResNet18 on GPU")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print("✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        print("\n" + "="*60)
        print("✅ ALL GPU TESTS PASSED!")
        print("="*60)
        print("\nYour environment is ready for ACT training with GPU acceleration.")
        
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        sys.exit(1)
        
else:
    print("\n" + "="*60)
    print("⚠️  WARNING: CUDA is not available!")
    print("="*60)
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU detected")
    print("2. CUDA drivers not installed")
    print("3. PyTorch installed without CUDA support")
    print("\nTraining will run on CPU (very slow).")
    sys.exit(1)
