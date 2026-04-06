'''
# rule of thumb for adjusting model complexity based on task difficulty
# (like adjusting a kid's drawing lessons based on the complexity of what they need to draw)
POLICY_CONFIG = {
    'enc_layers': 5,        # +1 more "seeing" lesson  
    'dec_layers': 8,        # +1 more "doing" lesson
    'hidden_dim': 512,      # Keep same "brain size"
    'dim_feedforward': 3200, # Keep same "thinking power"
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

# robot port names
ROBOT_PORTS = {
     'leader': '/dev/ttyUSB1',  # Leader arm (adjust if needed)
    'follower': '/dev/ttyUSB0'  # Follower arm
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 60,  # Task8 uses 60-frame episodes (changed from 400)
    'state_dim': 6,
    'action_dim': 6,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0  # /dev/video0
}


# policy config
POLICY_CONFIG = {
    'lr': 5e-5,
    'device': device,
    'num_queries': 5,  # Reduced from 10 for more detailed pick behavior
    'kl_weight': 100,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 5e-5,
    'backbone': 'resnet34',
    'enc_layers': 5,
    'dec_layers': 8,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 10000,
    'batch_size_val': 8,  # Reduced from 32 for 11GB GPU
    'batch_size_train': 8,  # Reduced from 32 for 11GB GPU
    'eval_ckpt_name': 'policy_best_epoch_9883_val_0.0260.ckpt',  # Original working task7c model
    'checkpoint_dir': CHECKPOINT_DIR
}