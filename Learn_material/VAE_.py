import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from scipy import stats

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SequenceVAE(nn.Module):
    def __init__(self, state_dim=6, action_dim=6, seq_len=10, hidden_dim=128, latent_dim=32):
        """
        A VAE for encoding and decoding action sequences
        
        Args:
            state_dim: Dimension of the robot state
            action_dim: Dimension of the robot actions
            seq_len: Length of action sequences to predict
            hidden_dim: Hidden dimension size
            latent_dim: Latent space dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder for state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Encoder for action sequence
        self.action_encoder = nn.GRU( # gated recurrent unit simulating transformer
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Combine state and action sequence encodings
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # action and pos projection 
            nn.ReLU()
        )
        
        # Mean and log variance for the latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder initial state from latent and robot state
        self.decoder_init = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GRU for generating action sequence
        self.decoder_rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # For auto-regressive decoding, we need an initial action
        self.initial_action = nn.Parameter(torch.zeros(1, action_dim))
    
    def encode(self, state, action_seq):
        """Encode state and action sequence to latent distribution parameters"""
        # Encode state
        state_encoding = self.state_encoder(state)
        
        # Encode action sequence with RNN
        _, action_encoding = self.action_encoder(action_seq)
        action_encoding = action_encoding[-1]  # Take the last layer's hidden state
        
        # Combine state and action encodings
        combined = torch.cat([state_encoding, action_encoding], dim=1)
        hidden = self.combined_encoder(combined)
        
        # Get latent distribution parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution using the reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, state, z, teacher_forcing_ratio=0.5, target_seq=None):
        """
        Decode state and latent vector to action sequence
        
        Args:
            state: Current robot state
            z: Latent vector
            teacher_forcing_ratio: Probability of using teacher forcing
            target_seq: Target sequence for teacher forcing (only used during training)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Initialize hidden state from latent vector and robot state
        hidden = self.decoder_init(torch.cat([z, state], dim=1))
        hidden = hidden.unsqueeze(0)  # Add layer dimension
        
        # Start with initial action (learned or zeros)
        decoder_input = self.initial_action.expand(batch_size, -1).unsqueeze(1)
        
        outputs = []
        
        # Generate sequence auto-regressively
        for t in range(self.seq_len):
            # Run one step of RNN
            output, hidden = self.decoder_rnn(decoder_input, hidden)
            
            # Predict action
            action = self.action_head(output.squeeze(1))
            outputs.append(action)
            
            # Teacher forcing: use ground truth or predicted action as next input
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1]  # (batch, 1, action_dim)
            else:
                decoder_input = action.unsqueeze(1)  # (batch, 1, action_dim)
        
        # Stack outputs to form sequence
        action_seq = torch.stack(outputs, dim=1)  # (batch, seq_len, action_dim)
        return action_seq
    
    def forward(self, state, action_seq):
        """Full forward pass: encode, sample, decode"""
        mu, logvar = self.encode(state, action_seq)
        z = self.reparameterize(mu, logvar)
        action_seq_recon = self.decode(state, z, teacher_forcing_ratio=0.5, target_seq=action_seq)
        return action_seq_recon, mu, logvar
    
    def generate(self, state, num_samples=5):
        """Generate multiple action sequence samples for a given state"""
        batch_size = state.shape[0]
        state_batch = state.repeat(num_samples, 1)
        z_samples = torch.randn(num_samples * batch_size, self.latent_dim).to(state.device)
        return self.decode(state_batch, z_samples, teacher_forcing_ratio=0.0)

class SequenceActionDataset(Dataset):
    def __init__(self, data_dir, seq_len=10):
        """Dataset for robot state and action sequences"""
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.episode_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        
        # Pre-load all state-action sequences
        self.initial_states = []
        self.action_sequences = []
        
        if self.episode_files:
            sample_file = os.path.join(data_dir, self.episode_files[0])
            with h5py.File(sample_file, 'r') as f:
                print("\nHDF5 file structure:")
                
                def print_attrs(name, obj):
                    print(f" - {name}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"   Shape: {obj.shape}, Dtype: {obj.dtype}")
                
                f.visititems(print_attrs)
                print("\n")
        
        for ep_file in self.episode_files:
            file_path = os.path.join(data_dir, ep_file)
            with h5py.File(file_path, 'r') as f:
                # Extract robot states and actions
                qpos = f['/observations/qpos'][:]  # Shape: [T, state_dim]
                actions = f['/action'][:]  # Shape: [T, action_dim]
                
                # Create sequences
                for t in range(len(actions) - seq_len):
                    self.initial_states.append(qpos[t])
                    self.action_sequences.append(actions[t:t+seq_len])
        
        self.initial_states = np.array(self.initial_states)
        self.action_sequences = np.array(self.action_sequences)
        
        print(f"Loaded {len(self.initial_states)} state-action sequences from {len(self.episode_files)} episodes")
        print(f"State shape: {self.initial_states.shape}, Action sequences shape: {self.action_sequences.shape}")
    
    def __len__(self):
        return len(self.initial_states)
    
    def __getitem__(self, idx):
        return {
            'state': torch.tensor(self.initial_states[idx], dtype=torch.float32),
            'action_seq': torch.tensor(self.action_sequences[idx], dtype=torch.float32)
        }

def train_sequence_vae(model, data_loader, num_epochs=50, lr=1e-3, beta=1.0, device='cpu'):
    """Train the Sequence VAE model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get data
            state = batch['state'].to(device)
            action_seq = batch['action_seq'].to(device)
            
            # Forward pass
            action_seq_recon, mu, logvar = model(state, action_seq)
            
            # Calculate losses
            recon_loss = F.mse_loss(action_seq_recon, action_seq)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state.size(0)
            loss = recon_loss + beta * kl_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        # Average losses
        avg_loss = epoch_loss / len(data_loader)
        avg_recon_loss = recon_loss_total / len(data_loader)
        avg_kl_loss = kl_loss_total / len(data_loader)
        
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, "
              f"Recon Loss: {avg_recon_loss:.6f}, KL Loss: {avg_kl_loss:.6f}")
    
    return losses

def visualize_latent_distributions(model, data_loader, device, dims_to_show=4):
    """Visualize the distributions of specific latent dimensions"""
    latent_values = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting latent values"):
            state = batch['state'].to(device)
            action_seq = batch['action_seq'].to(device)
            
            mu, logvar = model.encode(state, action_seq)
            latent_values.append(mu.cpu().numpy())
    
    latent_values = np.concatenate(latent_values, axis=0)
    
    # Plot histograms of latent dimensions
    fig, axes = plt.subplots(dims_to_show, 1, figsize=(10, 3*dims_to_show))
    
    for i in range(dims_to_show):
        ax = axes[i]
        ax.hist(latent_values[:, i], bins=50, alpha=0.7, density=True)
        
        # Fit a normal distribution to the data
        mu, std = stats.norm.fit(latent_values[:, i])
        x = np.linspace(min(latent_values[:, i]), max(latent_values[:, i]), 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        
        ax.set_title(f'Latent Dimension {i+1} Distribution (μ={mu:.2f}, σ={std:.2f})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_distributions.png', dpi=300)
    plt.show()
    
    # Plot 2D scatter of the first two dimensions with Gaussian contours
    plt.figure(figsize=(10, 8))
    
    # First, create a scatter plot
    plt.scatter(latent_values[:, 0], latent_values[:, 1], alpha=0.6, s=5)
    
    # Calculate and plot contours of the fitted 2D Gaussian
    from scipy.stats import multivariate_normal
    from matplotlib.patches import Ellipse
    
    # Compute mean and covariance
    mean = np.mean(latent_values[:, :2], axis=0)
    cov = np.cov(latent_values[:, :2], rowvar=False)
    
    # Plot mean point
    plt.scatter(mean[0], mean[1], c='red', s=100, marker='x', label='Mean')
    
    # Plot covariance ellipses
    for n_std in [1, 2, 3]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                         edgecolor='k', facecolor='none', linestyle='--',
                         alpha=0.7, label=f'{n_std}σ')
        plt.gca().add_patch(ellipse)
    
    plt.title('First Two Latent Dimensions with Gaussian Contours')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.savefig('latent_2d_gaussian.png', dpi=300)
    plt.show()
    
    return latent_values

def visualize_sequence_predictions(model, dataset, device, num_states=3):
    """Visualize sequence predictions for a few examples"""
    # Sample a few states from the dataset
    indices = np.random.choice(len(dataset), num_states, replace=False)
    
    model.eval()
    
    # Create a figure for visualization
    fig, axes = plt.subplots(num_states, 1, figsize=(12, 5*num_states))
    if num_states == 1:
        axes = [axes]
    
    action_dim = dataset[0]['action_seq'].shape[1]
    
    for i, idx in enumerate(indices):
        # Get the state and target sequence
        data = dataset[idx]
        state = data['state'].unsqueeze(0).to(device)
        target_seq = data['action_seq'].unsqueeze(0).to(device)
        
        # Generate multiple action sequence samples
        with torch.no_grad():
            num_samples = 5
            samples = model.generate(state, num_samples=num_samples)
        
        ax = axes[i]
        
        # Plot the ground truth sequence
        for d in range(action_dim):
            ax.plot(range(model.seq_len), target_seq[0, :, d].cpu().numpy(), 
                    'k-', linewidth=2, label=f'Ground Truth Dim {d+1}' if i == 0 else None)
        
        # Plot the predicted sequences with different colors per sample
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
        for s in range(num_samples):
            for d in range(action_dim):
                ax.plot(range(model.seq_len), samples[s, :, d].cpu().numpy(), 
                        '--', color=colors[s], alpha=0.7, 
                        label=f'Sample {s+1}, Dim {d+1}' if i == 0 and d == 0 else None)
        
        ax.set_title(f'Example {i+1}: Ground Truth vs Predicted Sequences')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig('sequence_predictions.png', dpi=300)
    plt.show()

def main():
    # Set up data and model
    data_dir = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sequence length for our model
    seq_len = 10
    
    # Create dataset and dataloader
    dataset = SequenceActionDataset(data_dir, seq_len=seq_len)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create model
    state_dim = dataset.initial_states.shape[1] if dataset.initial_states.size > 0 else 6
    action_dim = dataset.action_sequences.shape[2] if dataset.action_sequences.size > 0 else 6
    
    model = SequenceVAE(
        state_dim=state_dim, 
        action_dim=action_dim,
        seq_len=seq_len,
        hidden_dim=128,
        latent_dim=32
    ).to(device)
    
    # Load model if it exists, otherwise train a new one
    model_path = 'sequence_vae_model.pth'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training a new Sequence VAE model...")
        losses = train_sequence_vae(model, data_loader, num_epochs=30, device=device)
        
        # Plot training losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.savefig('sequence_vae_training_loss.png', dpi=300)
        plt.show()
        
        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Visualize the latent space distributions
    print("Visualizing latent distributions...")
    latent_values = visualize_latent_distributions(model, data_loader, device, dims_to_show=4)
    
    # Visualize sequence predictions
    print("Visualizing sequence predictions...")
    visualize_sequence_predictions(model, dataset, device, num_states=3)
    
    print("Experiment complete! Check the visualizations to understand the stochastic nature of the Sequence VAE.")

if __name__ == "__main__":
    main()