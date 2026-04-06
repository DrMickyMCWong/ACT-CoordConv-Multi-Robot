# ACT with CoordConv Integration for RoArm and Alicia Robots

This repository contains an enhanced adaptation of [Action Chunking Transformer (ACT)](https://github.com/tonyzhaozh/act/tree/main) with **CoordConv integration** for improved spatial awareness in real-world robotic manipulation tasks. The implementation supports both RoArm and Alicia robot platforms.

## 🚀 Key Innovation: CoordConv Integration

### The Spatial Awareness Problem
Standard CNNs in ACT are translation invariant — they recognize **what** is in an image but lose precise **where** information. In simulation, your camera position, lighting, and object placement are perfectly controlled. In real environments, small variations in camera angle, object position, and lighting cause the policy to fail because the spatial coordinates are subtly wrong even when the object looks correct.

### Our Solution: Coordinate-Aware Vision
Our implementation integrates **CoordConv layers** as additional channels prior to the CNN ResNet34 backbone. CoordConv appends the (x, y) pixel coordinate channels directly to the input, giving the network explicit spatial awareness. This means the policy learns "grip at this precise location in this coordinate space" rather than "grip at whatever looks like the object."

For fixed camera setups and repeatable object placement, this coordinate-aware approach provides substantial improvements in real-world robotic manipulation tasks.

## 🤖 Supported Robot Platforms

### Alicia Robot
![Alicia Robot Demo](https://img.shields.io/badge/▶️-Watch%20Demo-red?style=for-the-badge)
> [📹 **View Alicia Robot Demo Video**](./alicia_ver.mp4)
> 
> *Advanced manipulation capabilities with dual-arm configuration*

### RoArm Robot  
![RoArm Robot Demo](https://img.shields.io/badge/▶️-Watch%20Demo-blue?style=for-the-badge)
> [📹 **View RoArm Robot Demo Video**](./roarm_version.mp4)
> 
> *Compact and versatile single-arm manipulation*

### Sim-to-Real Transfer
![Sim-to-Real Demo](https://img.shields.io/badge/▶️-Watch%20Demo-green?style=for-the-badge)
> [📹 **View Sim-to-Real Transfer Video**](./real_to_sim_alicia_ver.mp4)
> 
> *Demonstration of simulation to real-world transfer capabilities*

## 🔬 Current Development Status

- ✅ **CoordConv Integration**: Implemented and tested
- ✅ **Multi-robot Support**: RoArm and Alicia platforms
- 🔄 **Sim-to-Real Transfer**: Pending integration with CoordConv architecture
- 🔄 **Reactive Obstacle Avoidance**: Network implemented, integration in progress

The enhanced spatial awareness from CoordConv makes this particularly suitable for tasks requiring precise manipulation in constrained environments.


## 🛠️ Setup and Installation

### Prerequisites
- **Robot Descriptions**: Clone the robot description repository:
  ```bash
  git clone https://github.com/Synria-Robotics/Synria-Robot-Descriptions
  ```

### Environment Setup
```bash
conda create --name act python=3.9
conda activate act
conda install pytorch==1.13.1 torchvision==0.14.1
pip install -r requirements.txt
```

### Configuration
1. Update robot connection ports in `config/config.py` (`TASK_CONFIG`)
2. Set camera port (default: 0)
3. Ensure camera position remains fixed during data collection and evaluation
4. Update paths in configuration files to point to the cloned robot descriptions

## 📊 Usage

### Data Collection
```bash
python record_episodes.py --task <task_name> --num_episodes <num>
```
- Data stored in `data/<task_name>/`
- Audio cues: "Go" to start recording, "Stop" to end
- Recommended: 20-30 demonstrations for good performance

### Training with CoordConv
```bash
python train.py --task <task_name>
```
- Enhanced with coordinate-aware features
- Checkpoints saved in `checkpoints/<task_name>/`
- Typical training time: ~30 minutes on RTX 3080

### Policy Evaluation
```bash
# For Alicia robot
python evaluate_custom_alicia.py --task <task_name>

# For RoArm robot  
python evaluate.py --task <task_name>
```

## 🔍 Architecture Details

The CoordConv integration adds explicit spatial coordinate channels to the input, enhancing the network's ability to learn precise spatial relationships. This is particularly effective for:
- Fixed camera setups
- Repeatable object placement scenarios  
- Tasks requiring precise spatial manipulation

## 🚧 Future Work

- Complete sim-to-real transfer integration with CoordConv architecture
- Finalize reactive obstacle avoidance network integration
- Enhanced multi-modal sensor fusion capabilities

## 📦 Dependencies

- **[Synria Robot Descriptions](https://github.com/Synria-Robotics/Synria-Robot-Descriptions)**: Robot models and configurations for RoArm and Alicia platforms
- PyTorch 1.13.1+ with torchvision
- Standard robotics libraries (see `requirements.txt`)

## 📈 Performance & Results

Training performance with CoordConv integration shows improved spatial precision:
- **Data efficiency**: 20-30 demonstrations sufficient for complex tasks
- **Training time**: ~30 minutes on RTX 3080
- **Spatial accuracy**: Significant improvement in precise manipulation tasks
- **Real-world robustness**: Enhanced performance under lighting/camera variations

## 🔧 Technical Implementation

### CoordConv Architecture
- Additional coordinate channels (x, y) appended to input
- Integrated before ResNet34 CNN backbone  
- Provides explicit spatial awareness to the network
- Particularly effective for fixed camera configurations

### Depth Processing
The implementation includes advanced depth image processing:
```python
# Depth normalization pipeline
depth_mm_to_meters = depth.astype(np.float32) / 1000.0
depth_clipped = np.clip(depth_mm_to_meters, 0.2, 0.8)  # Manipulation range
depth_normalized = ((depth_clipped - 0.2) / 0.6 * 255.0).astype(np.uint8)
```

## 📝 Citation

If you use this work in your research, please cite:
```bibtex
@article{act_coordconv_2026,
  title={Action Chunking Transformer with CoordConv Integration for Robotic Manipulation},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome! Areas of particular interest:
- Sim-to-real transfer improvements
- Multi-modal sensor integration
- Obstacle avoidance network refinements

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
