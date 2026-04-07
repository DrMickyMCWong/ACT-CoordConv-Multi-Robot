import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set device environment variable first
os.environ['DEVICE'] = 'cuda'

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.nn import functional as F

# Import training utilities (after adding the parent directory to the path)
from training.utils import make_policy
from detr.models.detr_vae import reparametrize  # Keep this import for the reparametrize function

class CVAEVisualizer:
    def __init__(self, model_path, dataset_path):
        """Initialize the CVAE visualizer with model and dataset paths"""
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.camera_names = ['front']
        
        print(f"Initializing CVAEVisualizer with model: {model_path}")
        print(f"Dataset: {dataset_path}")
        print(f"Using device: {self.device}")
        
        # Load the model
        self.load_model()
        
    def load_model(self):
        """Load the trained DETR-VAE model using the approach from evaluate_custom.py"""
        try:
            from training.utils import make_policy
            
            print("Loading checkpoint:", self.model_path)
            
            # Create policy config based on evaluate_custom.py
            # This matches how it's done in the evaluation script
            policy_config = {
                'num_queries': 100,
                'kl_weight': 100,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': 5e-5,
                'backbone': 'resnet34',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': ['front'],
                'policy_class': 'ACT',
                'temporal_agg': False
            }
            
            # Create the policy first
            print("Creating policy using make_policy...")
            self.model = make_policy("ACT", policy_config)
            
            # Print model structure
            print(f"Model structure created with {sum(p.numel() for p in self.model.parameters())} parameters")
            
            # Load state dict directly - THIS IS THE KEY DIFFERENCE
            # In evaluate_custom.py, they load the model weights directly without accessing a nested key
            print("Loading model weights directly...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Debug checkpoint content
            if isinstance(checkpoint, dict) and len(checkpoint) < 10:
                print("Checkpoint content keys:", checkpoint.keys())
            
            # In evaluate_custom.py, they use load_state_dict directly on the checkpoint
            loading_status = self.model.load_state_dict(checkpoint)
            print(f"Loading status: {loading_status}")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    
    def load_episode_data(self, episode_idx=0):
        """Load a specific episode from the dataset"""
        episode_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.hdf5')]
        if not episode_files:
            raise ValueError(f"No HDF5 files found in {self.dataset_path}")
        
        episode_file = episode_files[episode_idx]
        episode_path = os.path.join(self.dataset_path, episode_file)
        
        print(f"Loading episode data from {episode_path}")
        
        with h5py.File(episode_path, 'r') as root:
            # Debug HDF5 structure
            print("HDF5 root keys:", list(root.keys()))
            print("Observations keys:", list(root['/observations'].keys()))
            print("Images keys:", list(root['/observations/images'].keys()))
            
            # Load images, robot states, and actions
            images = root['/observations/images/front'][:]
            qpos = root['/observations/qpos'][:]
            actions = root['/action'][:]  # Changed from '/actions' to '/action'
            
            print(f"Loaded shapes - Images: {images.shape}, Qpos: {qpos.shape}, Actions: {actions.shape}")
            
            # Create padding mask for actions (no padding in this case)
            is_pad = torch.zeros(actions.shape[0], dtype=torch.bool)
            
        return {
            'images': torch.from_numpy(images).float(),  # (seq_len, H, W, C)
            'qpos': torch.from_numpy(qpos).float(),      # (seq_len, qpos_dim)
            'actions': torch.from_numpy(actions).float(), # Using consistent key name
            'is_pad': is_pad                             # (seq_len)
        }
    
    def preprocess_batch(self, episode_data, frame_idx=0, seq_length=10):
        """Prepare a batch of data for the model"""
        # Get a segment of the episode starting from frame_idx
        end_idx = min(frame_idx + seq_length, episode_data['actions'].shape[0])
        
        # Extract the relevant sequence
        qpos = episode_data['qpos'][frame_idx].unsqueeze(0)  # (1, qpos_dim)
        images = episode_data['images'][frame_idx].unsqueeze(0)  # (1, H, W, C)
        
        # Convert image format from (H, W, C) to (C, H, W)
        images = images.permute(0, 3, 1, 2)
        
        # Add camera dimension expected by the model
        images = images.unsqueeze(1)  # (batch, cam, C, H, W)
        
        # Get action sequence
        actions = episode_data['actions'][frame_idx:end_idx].unsqueeze(0)  # (1, seq, action_dim)
        
        # Create padding mask (all False since we're not padding here)
        seq_len = end_idx - frame_idx
        is_pad = torch.zeros(1, seq_len, dtype=torch.bool)
        
        print(f"Preprocessed batch - Qpos: {qpos.shape}, Image: {images.shape}, Actions: {actions.shape}")
        
        return {
            'qpos': qpos.to(self.device),
            'image': images.to(self.device),
            'actions': actions.to(self.device),
            'is_pad': is_pad.to(self.device),
            'env_state': torch.zeros(1, 6).to(self.device)  # Placeholder if needed
        }
    
    def generate_multiple_action_sequences(self, batch, num_samples=5):
        """Generate multiple action sequences from the same input"""
        results = []
        latent_vectors = []
        
        with torch.no_grad():
            for i in range(num_samples):
                print(f"Generating sample {i+1}/{num_samples}")
                # Forward pass through the model
                actions_hat, is_pad_hat, latent_info = self.model(
                    batch['qpos'],
                    batch['image'],
                    batch['env_state'],
                    batch['actions'],
                    batch['is_pad']
                )
                
                # Store results
                results.append(actions_hat.cpu().numpy())
                
                # Store latent vectors
                mu, logvar = latent_info
                if mu is not None and logvar is not None:
                    latent = reparametrize(mu, logvar)
                    latent_vectors.append(latent.cpu().numpy())
        
        return results, latent_vectors
    
    def visualize_action_variations(self, action_sequences, latent_vectors, original_actions):
        """Visualize the variation in generated action sequences"""
        num_samples = len(action_sequences)
        action_dim = action_sequences[0].shape[-1]
        seq_length = action_sequences[0].shape[1]
        
        print(f"Visualizing variations across {num_samples} samples, {action_dim} action dimensions, {seq_length} time steps")
        
        # Plot the action sequences
        fig, axes = plt.subplots(action_dim, 1, figsize=(12, 3*action_dim))
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
        
        for dim in range(action_dim):
            ax = axes[dim] if action_dim > 1 else axes
            
            # Plot original actions in black
            ax.plot(original_actions[0, :, dim].cpu().numpy(), 
                    color='black', linewidth=2, label='Original')
            
            # Plot generated actions in different colors
            for i, seq in enumerate(action_sequences):
                ax.plot(seq[0, :, dim], color=colors[i], 
                        alpha=0.7, label=f'Sample {i+1}')
            
            ax.set_title(f'Action Dimension {dim+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action Value')
            
            if dim == 0:
                ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('action_variations.png', dpi=300)
        plt.show()
        
        # Visualize latent space if we have multiple latent vectors
        if latent_vectors and len(latent_vectors) > 1:
            # Convert to numpy array for easier manipulation
            latent_array = np.array(latent_vectors).squeeze()
            
            print(f"Latent vectors shape: {latent_array.shape}")
            
            # If latent dimension > 2, use PCA or t-SNE for visualization
            if latent_array.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latent_2d = pca.fit_transform(latent_array)
                print("Applied PCA to reduce latent dimensions to 2D")
            else:
                latent_2d = latent_array
            
            # Plot the latent vectors in 2D
            plt.figure(figsize=(8, 8))
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors)
            
            for i, (x, y) in enumerate(latent_2d):
                plt.annotate(f"{i+1}", (x, y), fontsize=12)
                
            plt.title("Latent Space Visualization")
            plt.xlabel("Latent Dimension 1")
            plt.ylabel("Latent Dimension 2")
            plt.grid(True, alpha=0.3)
            plt.savefig('latent_space.png', dpi=300)
            plt.show()
    
    def visualize_image_with_actions(self, image, actions, frame_idx):
        """Visualize the robot image with overlaid action predictions"""
        # Convert tensor to numpy array for visualization
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8 for visualization
        image_np = (image_np * 255).astype(np.uint8)
        
        # Create a figure to show the image with action arrows
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        plt.title(f"Frame {frame_idx} with Action Predictions")
        
        # Extract the position and orientation actions (assuming first 3 are position, next 3 are orientation)
        pos_actions = actions[:, :3]  # First 3 dimensions often represent position
        
        # Draw arrows for the first few predicted actions
        for i in range(min(5, len(pos_actions))):
            # Scale actions for visualization
            arrow_length = np.linalg.norm(pos_actions[i]) * 50  # Scale factor
            
            # Normalize the direction vector
            direction = pos_actions[i] / np.linalg.norm(pos_actions[i]) if np.linalg.norm(pos_actions[i]) > 0 else np.zeros(3)
            
            # Start point (center of image)
            start_x, start_y = image_np.shape[1] // 2, image_np.shape[0] // 2
            
            # End point (using only x, y components for 2D visualization)
            end_x = start_x + int(direction[0] * arrow_length)
            end_y = start_y + int(direction[1] * arrow_length)
            
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                     head_width=10, head_length=10, fc=f'C{i}', ec=f'C{i}', 
                     length_includes_head=True, alpha=0.7)
        
        plt.axis('off')
        plt.savefig(f'frame_{frame_idx}_actions.png', dpi=300)
        plt.show()

    def run_experiment(self, episode_idx=0, frame_idx=0, num_samples=5):
        """Run the full experiment with visualization"""
        print("Loading episode data...")
        episode_data = self.load_episode_data(episode_idx)
        
        print(f"Preparing batch at frame {frame_idx}...")
        batch = self.preprocess_batch(episode_data, frame_idx)
        
        print(f"Generating {num_samples} action sequences...")
        action_sequences, latent_vectors = self.generate_multiple_action_sequences(batch, num_samples)
        
        print("Visualizing action variations...")
        self.visualize_action_variations(action_sequences, latent_vectors, batch['actions'])
        
        print("Visualizing image with predicted actions...")
        self.visualize_image_with_actions(batch['image'], action_sequences[0][0], frame_idx)
        
        print("Experiment completed!")

def main():
    # Set paths to your actual model
    model_path = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\checkpoints\task1\policy_best_epoch_4904_val_0.2318.ckpt'
    dataset_path = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task1'
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize and run the experiment
    print("Initializing visualizer with real model and data...")
    visualizer = CVAEVisualizer(model_path, dataset_path)
    
    # Run for multiple episodes to see variations
    for episode_idx in range(1):  # Start with just one episode
        for frame_idx in [0]:  # Start with just one frame
            print(f"\nRunning experiment for episode {episode_idx}, starting at frame {frame_idx}")
            visualizer.run_experiment(episode_idx=episode_idx, frame_idx=frame_idx, num_samples=5)
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()