import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import torchvision
import cv2
# Set device environment variable
os.environ['DEVICE'] = 'cuda'  # or 'cuda' if you have a GPU

def load_image_from_hdf5(dataset_path, frame_idx=150, camera_name='front'):
    """Load a specific frame from an HDF5 file"""
    with h5py.File(dataset_path, 'r') as root:
        images = root[f'/observations/images/{camera_name}'][()]
        if frame_idx < len(images):
            return images[frame_idx]
        else:
            print(f"Frame index {frame_idx} out of range. Using first frame.")
            return images[0]

def preprocess_image(image):
    """Convert numpy image to tensor and normalize for ResNet"""
    # Convert to PIL image first (if it's a numpy array)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply standard ResNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def get_resnet_feature_extractor():
    """Create a ResNet18 feature extractor that returns intermediate features"""
    # Load a pretrained ResNet
    model = models.resnet18(pretrained=True)
    
    # Create a feature extractor that returns outputs from different layers
    layers = {
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4
    }
    
    return model, layers

def extract_features(model, layers, x):
    """Extract features from different layers of ResNet"""
    # Initial layers
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    
    # Extract features from each layer
    features = {}
    for name, layer in layers.items():
        x = layer(x)
        features[name] = x
    
    return features

def visualize_feature_maps(features, original_image, max_features=16):
    """Visualize the feature maps from different layers"""
    plt.figure(figsize=(15, 8))
    
    # Show original image
    plt.subplot(1, len(features) + 1, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # For each layer's feature maps
    for layer_idx, (layer_name, feature_map) in enumerate(features.items()):
        # Get the feature map (first image in batch)
        feature_tensor = feature_map[0].detach().cpu()
        
        # Create a grid of feature maps
        grid_size = min(max_features, feature_tensor.size(0))
        rows = int(np.sqrt(grid_size))
        cols = int(np.ceil(grid_size / rows))
        
        # Create a subplot for this layer
        plt.subplot(1, len(features) + 1, layer_idx + 2)
        plt.title(f'Layer {layer_name} Features')
        
        # Combine feature maps into a grid
        grid = torch.zeros((grid_size, feature_tensor.size(1), feature_tensor.size(2)))
        for i in range(grid_size):
            grid[i] = feature_tensor[i]
        
        # Normalize for better visualization
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-5)
        
        # Create a grid image
        grid_img = torchvision.utils.make_grid(
            grid.unsqueeze(1), nrow=cols, padding=1, normalize=False
        )
        
        # Convert to numpy and transpose for plotting
        grid_img = grid_img.numpy().transpose((1, 2, 0))
        
        # Plot the grid
        plt.imshow(grid_img[:, :, 0], cmap='viridis')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=300)
    plt.show()
    
    
def visualize_top_channels(image, feature_map, layer_name, num_channels=5):
    """Visualize the top-activating channels and overlay them on the original image"""
    # Get feature map
    feature_tensor = feature_map[0].detach().cpu()
    
    # Find channels with highest activation
    channel_max_activations = feature_tensor.max(dim=1)[0].max(dim=1)[0]
    top_channels = torch.argsort(channel_max_activations, descending=True)[:num_channels]
    
    plt.figure(figsize=(15, 3*num_channels))
    
    for i, channel_idx in enumerate(top_channels):
        channel = feature_tensor[channel_idx].numpy()
        
        # Resize for visualization
        resized_channel = cv2.resize(channel, (image.shape[1], image.shape[0]))
        
        # Normalize
        resized_channel = (resized_channel - resized_channel.min()) / (resized_channel.max() - resized_channel.min() + 1e-5)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((resized_channel * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = 0.7 * image + 0.3 * heatmap
        
        plt.subplot(num_channels, 2, i*2+1)
        plt.imshow(resized_channel, cmap='viridis')
        plt.title(f'Channel {channel_idx} Feature')
        plt.axis('off')
        
        plt.subplot(num_channels, 2, i*2+2)
        plt.imshow(overlay.astype(np.uint8))
        plt.title(f'Channel {channel_idx} Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{layer_name}_top_channels.png', dpi=300)
    plt.show()    

def overlay_activations_on_image(image, feature_map, layer_name):
    # Get feature map for a specific layer
    feature = feature_map[0].detach().cpu().sum(dim=0)  # Sum across channels
    
    # Normalize for visualization
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    
    # Resize to match original image
    feature_np = feature.numpy()
    resized_feature = cv2.resize(feature_np, (image.shape[1], image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap((resized_feature * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlay = 0.7 * image + 0.3 * heatmap
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay.astype(np.uint8))
    plt.title(f'{layer_name} Activation Overlay')
    plt.axis('off')
    
    plt.savefig(f'{layer_name}_overlay.png', dpi=300)
    plt.show()


def examine_channel_consistency(channel_idx=398, num_episodes=5, frames_per_episode=3):
    """
    Examine if a specific channel consistently activates for the same objects
    across multiple episodes and frames.
    
    Args:
        channel_idx: The channel to track (e.g., 398 for end effector)
        num_episodes: Number of episodes to sample
        frames_per_episode: Number of frames to sample per episode
    """
    # Load an image from your dataset
    data_dir = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task1'
    episode_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    
    if len(episode_files) == 0:
        print("No episodes found!")
        return
        
    # Limit to available episodes
    num_episodes = min(num_episodes, len(episode_files))
    
    # Model setup
    model, layers = get_resnet_feature_extractor()
    model.eval()
    
    # For storing results
    all_activations = []
    all_images = []
    
    # Setup the figure
    fig_rows = num_episodes
    fig_cols = frames_per_episode
    plt.figure(figsize=(5*fig_cols, 5*fig_rows))
    
    # For each episode
    for ep_idx in range(num_episodes):
        episode_file = episode_files[ep_idx]
        dataset_path = os.path.join(data_dir, episode_file)
        
        # Get the total number of frames
        with h5py.File(dataset_path, 'r') as root:
            total_frames = len(root[f'/observations/images/front'])
        
        # Calculate frame indices to sample (beginning, middle, end)
        if frames_per_episode == 3:
            frame_indices = [int(total_frames * 0.1), 
                             int(total_frames * 0.5), 
                             int(total_frames * 0.9)]
        else:
            # Evenly spaced frames
            frame_indices = [int(i * total_frames / frames_per_episode) 
                             for i in range(frames_per_episode)]
        
        # Process each frame
        for frame_idx, frame_num in enumerate(frame_indices):
            # Load and preprocess image
            image = load_image_from_hdf5(dataset_path, frame_idx=frame_num)
            image_tensor = preprocess_image(image)
            
            # Extract features
            features = extract_features(model, layers, image_tensor)
            
            # Get activation for this specific channel
            channel_activation = features['layer4'][0][channel_idx].detach().cpu()
            
            # Resize for visualization
            resized_activation = cv2.resize(channel_activation.numpy(), 
                                           (image.shape[1], image.shape[0]))
            
            # Normalize for visualization
            norm_activation = (resized_activation - resized_activation.min()) / \
                             (resized_activation.max() - resized_activation.min() + 1e-5)
            
            # Create heatmap
            heatmap = cv2.applyColorMap((norm_activation * 255).astype(np.uint8), 
                                        cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay on image
            overlay = 0.7 * image + 0.3 * heatmap
            
            # Store for later analysis
            all_activations.append(channel_activation)
            all_images.append(image)
            
            # Plot
            plt_idx = ep_idx * fig_cols + frame_idx + 1
            plt.subplot(fig_rows, fig_cols, plt_idx)
            plt.imshow(overlay.astype(np.uint8))
            plt.title(f"Ep {ep_idx+1}, Frame {frame_num}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'channel_{channel_idx}_consistency.png', dpi=300)
    plt.show()
    
    # Calculate consistency metrics
    if len(all_activations) > 1:
        # Compute activation statistics
        activation_max_positions = []
        for act in all_activations:
            # Find position of maximum activation
            max_pos = np.unravel_index(torch.argmax(act).item(), act.shape)
            activation_max_positions.append(max_pos)
        
        # Calculate consistency score (how often max activation is in same region)
        consistency = analyze_activation_consistency(activation_max_positions)
        print(f"Channel {channel_idx} consistency score: {consistency:.2f}/1.00")
        
        # Optionally: Compare multiple important channels
        if channel_idx == 398:  # If we're looking at the end effector channel
            print("Analyzing additional channels for comparison...")
            for compare_channel in [367, 475, 472]:  # Other channels you identified
                examine_channel_consistency(channel_idx=compare_channel, 
                                           num_episodes=2, frames_per_episode=2)

def analyze_activation_consistency(positions):
    """
    Calculate a consistency score based on activation positions
    Higher score means more consistent positioning
    """
    if not positions:
        return 0
        
    # Convert positions to numpy for easier manipulation
    pos_array = np.array(positions)
    
    # Calculate mean position
    mean_pos = np.mean(pos_array, axis=0)
    
    # Calculate average distance from mean (normalized by feature map size)
    distances = np.sqrt(np.sum((pos_array - mean_pos)**2, axis=1))
    
    # Normalize by feature map diagonal (√(7²+7²) for 7x7 feature map)
    max_distance = np.sqrt(7**2 + 7**2)
    avg_distance = np.mean(distances)
    
    # Convert to consistency score (1 = perfect consistency, 0 = random)
    consistency = 1.0 - (avg_distance / max_distance)
    
    return consistency


def visualize_more_channels(image, feature_map, num_channels=20, start_idx=0):
    """Visualize more channels to find brick detectors"""
    feature_tensor = feature_map[0].detach().cpu()
    
    plt.figure(figsize=(15, 30))
    
    for i in range(num_channels):
        channel_idx = start_idx + i
        if channel_idx >= feature_tensor.shape[0]:
            break
            
        # Get channel activation
        channel = feature_tensor[channel_idx].numpy()
        
        # Resize and normalize for visualization
        resized_channel = cv2.resize(channel, (image.shape[1], image.shape[0]))
        norm_channel = (resized_channel - resized_channel.min()) / \
                      (resized_channel.max() - resized_channel.min() + 1e-5)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((norm_channel * 255).astype(np.uint8), 
                                   cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on image
        overlay = 0.7 * image + 0.3 * heatmap
        
        # Plot side by side
        plt.subplot(num_channels, 2, i*2+1)
        plt.imshow(norm_channel, cmap='viridis')
        plt.title(f'Channel {channel_idx} Feature')
        plt.axis('off')
        
        plt.subplot(num_channels, 2, i*2+2)
        plt.imshow(overlay.astype(np.uint8))
        plt.title(f'Channel {channel_idx} Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'more_channels_{start_idx}.png', dpi=300)
    plt.show()


def main():
    # Load an image from your dataset
    data_dir = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task1'
    episode_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    
    if not episode_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    # Use the first episode file
    dataset_path = os.path.join(data_dir, episode_files[0])
    print(f"Loading image from: {dataset_path}")
    
    # Load an image from the middle of the episode
    image = load_image_from_hdf5(dataset_path, frame_idx=150)
    print(f"Image shape: {image.shape}")
    
    # Preprocess the image for the model
    image_tensor = preprocess_image(image)
    
    # Initialize the ResNet model and get layers
    model, layers = get_resnet_feature_extractor()
    model.eval()  # Set to evaluation mode
    
    # Extract features
    features = extract_features(model, layers, image_tensor)
    
    # Visualize the feature maps
    visualize_feature_maps(features, image)
    
    # Print feature map shapes
    print("\nFeature extraction pipeline:")
    print("1. Input image shape:", image.shape)
    print("2. Preprocessed tensor shape:", image_tensor.shape)
    
    for layer_name, feature_map in features.items():
        print(f"3. {layer_name} output shape: {feature_map.shape}")
    
    # Extra: Save one full feature map for detailed inspection
    layer_name = list(features.keys())[-1]  # Last layer
    feature = features[layer_name][0][0].detach().cpu().numpy()  # First channel of first batch
    
    plt.figure(figsize=(8, 8))
    plt.imshow(feature, cmap='viridis')
    plt.colorbar(label='Activation')
    plt.title(f'Detailed view of one {layer_name} feature')
    plt.savefig('detailed_feature.png', dpi=300)
    plt.show()

    # Then call this for layer4
    layer_name = 'layer4'
    overlay_activations_on_image(image, features[layer_name], layer_name)
    
    # Call the top channels visualization
    visualize_top_channels(image, features['layer4'], 'layer4')
    
    # Check consistency of channel 398 (end effector detector)
    examine_channel_consistency(channel_idx=398, num_episodes=5, frames_per_episode=3)
    
    # Call with different starting indices to explore more channels
    visualize_more_channels(image, features['layer4'], num_channels=20, start_idx=0)
    visualize_more_channels(image, features['layer4'], num_channels=20, start_idx=20)
    visualize_more_channels(image, features['layer4'], num_channels=20, start_idx=40)


if __name__ == "__main__":
    main()