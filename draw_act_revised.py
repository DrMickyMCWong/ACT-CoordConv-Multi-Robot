import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def draw_act_model_dataflow():
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Define colors
    encoder_color = '#E8F4FD'  # Light blue
    decoder_color = '#FFF2CC'  # Light yellow
    latent_color = '#E1D5E7'   # Light purple
    data_color = '#F8CECC'     # Light red
    output_color = '#D5E8D4'   # Light green
    collection_color = '#FFE6CC'  # Light orange
    
    # Title
    ax.text(10, 17.5, 'ACT-Shaka: Data Collection → Model Training → Inference with Tensor Dimensions', 
            fontsize=16, ha='center', weight='bold')
    
    # DATA COLLECTION PHASE (Top section)
    ax.text(10, 16.8, 'Phase 1: Teleoperation Data Collection (record_episodes.py)', 
            fontsize=14, ha='center', weight='bold', color='darkorange')
    
    # Leader robot
    leader_box = FancyBboxPatch((0.5, 15.8), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=collection_color, edgecolor='darkorange')
    ax.add_patch(leader_box)
    ax.text(2, 16.2, 'Leader Robot\n(Human Control)\nPWM Actions: [5]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Camera
    camera_box = FancyBboxPatch((4.5, 15.8), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=collection_color, edgecolor='darkorange')
    ax.add_patch(camera_box)
    ax.text(6, 16.2, 'Camera\nRGB Images: [480, 640, 3]\nCropped & Resized', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Follower robot
    follower_box = FancyBboxPatch((8.5, 15.8), 3, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=collection_color, edgecolor='darkorange')
    ax.add_patch(follower_box)
    ax.text(10, 16.2, 'Follower Robot\nState: qpos [5], qvel [5]\nPWM → Radians', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # HDF5 data
    hdf5_box = FancyBboxPatch((12.5, 15.8), 3.5, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=collection_color, edgecolor='darkorange')
    ax.add_patch(hdf5_box)
    ax.text(14.25, 16.2, 'HDF5 Episodes\nqpos: [300, 5]\nqvel: [300, 5]\nimages: [300, 480, 640, 3]\naction: [300, 5]', 
            ha='center', va='center', fontsize=8, weight='bold')

    # TRAINING PHASE
    ax.text(10, 14.8, 'Phase 2: Model Training (train.py) - Tensor Dimensions', 
            fontsize=14, ha='center', weight='bold', color='blue')
    
    # Data Loading with dimensions
    ax.text(10, 14.3, 'DataLoader: batch_size=8, episodes=80% train/20% val', 
            fontsize=10, ha='center', style='italic')
    
    # INPUT DATA (Training inputs from HDF5)
    # Current joint positions
    qpos_box = FancyBboxPatch((0.5, 13), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=data_color, edgecolor='black')
    ax.add_patch(qpos_box)
    ax.text(2, 13.5, 'Current State (qpos)\n[batch=8, state_dim=5]\nNormalized: (x-μ)/σ', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Camera images
    img_box = FancyBboxPatch((4.5, 13), 3, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=data_color, edgecolor='black')
    ax.add_patch(img_box)
    ax.text(6, 13.5, 'Camera Images\n[batch=8, cameras=1, C=3, H=480, W=640]\nNormalized: /255.0 + ImageNet', 
            ha='center', va='center', fontsize=8, weight='bold')
    
    # Future actions (ground truth from leader)
    actions_box = FancyBboxPatch((8.5, 13), 3.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=data_color, edgecolor='black')
    ax.add_patch(actions_box)
    ax.text(10.25, 13.5, 'Target Actions (Leader)\n[batch=8, chunk=100, action_dim=5]\nNormalized: (x-μ)/σ', 
            ha='center', va='center', fontsize=8, weight='bold')
    
    # Padding mask
    pad_box = FancyBboxPatch((13, 13), 2.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=data_color, edgecolor='black')
    ax.add_patch(pad_box)
    ax.text(14.25, 13.5, 'Padding Mask\n[batch=8, chunk=100]\nBool tensor', 
            ha='center', va='center', fontsize=8, weight='bold')

    # VISION BACKBONE
    ax.text(3, 11.8, 'Vision Backbone (ResNet18)', 
            fontsize=12, ha='center', weight='bold', color='green')
    
    vision_box = FancyBboxPatch((0.5, 10.5), 5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=output_color, edgecolor='green')
    ax.add_patch(vision_box)
    ax.text(3, 11, 'CNN Feature Extraction\nInput: [8, 1, 3, 480, 640]\nOutput: [8*1, 512, 15, 20]', 
            ha='center', va='center', fontsize=9, weight='bold')

    # TRANSFORMER ENCODER-DECODER
    ax.text(13, 11.8, 'ACT Transformer', 
            fontsize=12, ha='center', weight='bold', color='purple')
    
    # Encoder
    encoder_box = FancyBboxPatch((10, 10.5), 3, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=encoder_color, edgecolor='blue')
    ax.add_patch(encoder_box)
    ax.text(11.5, 11, 'Transformer Encoder\nLayers: 4, Heads: 8\nInput: [8, 300, 512]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Decoder (CVAE)
    decoder_box = FancyBboxPatch((14, 10.5), 3.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=decoder_color, edgecolor='orange')
    ax.add_patch(decoder_box)
    ax.text(15.75, 11, 'CVAE Decoder\nLayers: 7, Queries: 100\nμ, σ: [8, 256], Actions: [8, 100, 5]', 
            ha='center', va='center', fontsize=8, weight='bold')

    # LOSS COMPUTATION
    ax.text(10, 9.3, 'Loss Computation', 
            fontsize=12, ha='center', weight='bold', color='red')
    
    loss_box = FancyBboxPatch((6, 8.5), 8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#FFCCCC', edgecolor='red')
    ax.add_patch(loss_box)
    ax.text(10, 8.9, 'L1 Loss: ||predicted_actions - target_actions|| + KL_weight * KL_div(μ, σ)\nWeights: L1=1.0, KL=10.0', 
            ha='center', va='center', fontsize=9, weight='bold')

    # INFERENCE PHASE
    ax.text(10, 7.5, 'Phase 3: Inference/Deployment', 
            fontsize=14, ha='center', weight='bold', color='darkgreen')
    
    # Real-time inputs
    rt_qpos_box = FancyBboxPatch((1, 6.2), 2.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E6F3FF', edgecolor='darkblue')
    ax.add_patch(rt_qpos_box)
    ax.text(2.25, 6.6, 'Current qpos\n[1, 5]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    rt_img_box = FancyBboxPatch((4.5, 6.2), 2.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#E6F3FF', edgecolor='darkblue')
    ax.add_patch(rt_img_box)
    ax.text(5.75, 6.6, 'Current image\n[1, 1, 3, 480, 640]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Policy forward pass
    policy_box = FancyBboxPatch((8, 6.2), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#E6FFE6', edgecolor='darkgreen')
    ax.add_patch(policy_box)
    ax.text(10, 6.6, 'ACT Policy Forward\nOutput: [1, 100, 5] → Take first action [1, 5]', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Robot control
    control_box = FancyBboxPatch((13.5, 6.2), 2.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFE6E6', edgecolor='darkred')
    ax.add_patch(control_box)
    ax.text(14.75, 6.6, 'Robot Control\nDenormalize & PWM', 
            ha='center', va='center', fontsize=9, weight='bold')

    # ARROWS - Data flow connections
    # Collection to training
    ax.arrow(14.25, 15.5, 0, -1.7, head_width=0.2, head_length=0.1, fc='orange', ec='orange')
    
    # Training data flow
    ax.arrow(2, 12.7, 0, -1.5, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    ax.arrow(6, 12.7, -2.5, -1.5, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    ax.arrow(10.25, 12.7, 1, -1.5, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    
    # Encoder to decoder
    ax.arrow(13, 11, 0.8, 0, head_width=0.15, head_length=0.1, fc='purple', ec='purple')
    
    # To loss
    ax.arrow(15.75, 10.2, -5, -1.2, head_width=0.15, head_length=0.1, fc='red', ec='red')
    
    # Training to inference
    ax.arrow(10, 8.2, 0, -1.2, head_width=0.15, head_length=0.1, fc='darkgreen', ec='darkgreen')
    
    # Inference flow
    ax.arrow(3.5, 6.6, 4, 0, head_width=0.1, head_length=0.1, fc='darkgreen', ec='darkgreen')
    ax.arrow(12, 6.6, 1.3, 0, head_width=0.1, head_length=0.1, fc='darkgreen', ec='darkgreen')

    # KEY INSIGHTS
    ax.text(10, 5.2, 'Key Architecture Insights', 
            fontsize=12, ha='center', weight='bold', color='black')
    
    insights_text = """
    • Teleoperation: Leader-follower setup with synchronized data collection
    • CVAE Training: Learns conditional action distribution given state+image
    • Chunked Prediction: Predicts 100 future actions, executes first one
    • Normalization: Critical for stable training (qpos, actions normalized by dataset stats)
    • Vision: ResNet18 backbone → 512D features → Transformer input
    • Multi-modal: Combines proprioceptive (qpos) + visual (RGB) → action sequences
    """
    
    ax.text(10, 3.8, insights_text, 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))

    plt.tight_layout()
    plt.savefig('act_shaka_model_dataflow_with_dimensions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the function
draw_act_model_dataflow()
