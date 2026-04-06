#!/bin/bash
# Quick setup script to run after conda environment is created
# Run this with: bash post_install_setup.sh

echo "=================================================="
echo "ACT Post-Installation Setup"
echo "=================================================="
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "ACT"; then
    echo "❌ Error: ACT conda environment not found!"
    echo "Please create it first with:"
    echo "  conda env create -f act-main/conda_env.yaml"
    exit 1
fi

echo "✓ ACT conda environment found"
echo ""

# Activate environment and run GPU check
echo "=================================================="
echo "Step 1: Verifying GPU Functionality"
echo "=================================================="
echo ""

eval "$(conda shell.bash hook)"
conda activate ACT

python verify_gpu.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU verification passed!"
else
    echo ""
    echo "❌ GPU verification failed!"
    echo "Check CUDA installation and drivers."
    exit 1
fi

echo ""
echo "=================================================="
echo "Step 2: Configuration Summary"
echo "=================================================="
echo ""
echo "Converted dataset: /home/hk/Documents/ACT_Shaka/data/task7c_new/"
echo "Number of episodes: $(ls /home/hk/Documents/ACT_Shaka/data/task7c_new/ | wc -l)"
echo ""
echo "Next steps:"
echo "1. Update config/config.py with Linux paths"
echo "2. Run training: python train.py --task task7c_new"
echo ""
echo "=================================================="
echo "Setup complete! Environment is ready."
echo "=================================================="
