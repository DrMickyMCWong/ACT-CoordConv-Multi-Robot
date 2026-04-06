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
     'leader': 'COM6',  # Leader arm on COM6
    'follower': 'COM5'  # Follower arm on COM5
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 400,
    'state_dim': 6,
    'action_dim': 6,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front', 'side'],
    'camera_port': [0, 1]
}


# policy config
POLICY_CONFIG = {
    'lr': 5e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 50,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 5e-5,
    'backbone': 'resnet34',
    'temporal_weight': 0.3,        # Start lower - dynamic loss is often stronger
    'use_dynamic_temporal': True,
    'enc_layers': 6,
    'dec_layers': 8,
    'nheads': 8,
    'camera_names': ['front', 'side'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 5000,
    'batch_size_val': 32,
    'batch_size_train': 32,
    'eval_ckpt_name': 'policy_best_epoch_3430_val_0.1507.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}