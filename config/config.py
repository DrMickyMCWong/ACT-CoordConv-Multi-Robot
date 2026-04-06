'''
# rule of thumb for adjusting model complexity based on task difficulty
# (like adjusting a kid's drawing lessons based on the complexity of what they need to draw)
POLICY_CONFIG = {
    'enc_layers': 5,        # +1 more "seeing" lesson  
    'dec_layers': 8,        # +1 more "doing" lesson
    'hidden_dim': 512,      # Keep same "brain size"
    'dim_feedforward': 3200, # Keep same "thinking power"
    # ... other settings
# - Backbone: ResNet34 (compatible with 512-channel checkpoint)

POLICY_CONFIG = {
    'enc_layers': 4,        # Match checkpoint (was 5)
    'dec_layers': 7,        # Match checkpoint (was 8) 
    'hidden_dim': 512,      # Keep same "brain size"
    'dim_feedforward': 3200, # Keep same "thinking power"
    'num_queries': 100,     # Match checkpoint (was 5)
    # ... other settings
}

'''
import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '/home/hk/Documents/ACT_Shaka/data'

# checkpoint directory
CHECKPOINT_DIR = '/home/hk/Documents/ACT_Shaka/checkpoints'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names - Updated for Alicia SDK
ROBOT_PORTS = {
     'leader': '/dev/ttyACM1',  # Leader arm (corrected typo)
    'follower': '/dev/ttyACM2'  # Follower arm
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 1000,
    'state_dim': 7,
    'action_dim': 7,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'state_dim': 7,
    'action_dim': 7,
    'num_queries': 100,
    'kl_weight': 10.0,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet34',
    'enc_layers': 5,
    'dec_layers': 8,
    'nheads': 8,
    'latent_dim': 32,
    'vae_encoder_layers': 4,
    'dropout': 0.1,
    'pre_norm': False,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': False
}


# training config
TRAIN_CONFIG = {
    'seed': 1000,
    'num_epochs': 100000,
    'max_steps': 60000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_best_epoch_7576_val_0.1567.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR,
    'eval_freq': 2000,
    'log_freq': 100,
    'save_freq': 4000,
    'grad_clip_norm': 10.0
}