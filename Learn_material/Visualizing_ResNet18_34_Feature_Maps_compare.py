import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from PIL import Image
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

def get_resnet_model(model_name='resnet18'):
    """Get either ResNet18 or ResNet34 model"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
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

def visualize_top_channels(image, feature_map, layer_name, model_name, num_channels=5):
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
        plt.title(f'{model_name} Channel {channel_idx} Feature')
        plt.axis('off')
        
        plt.subplot(num_channels, 2, i*2+2)
        plt.imshow(overlay.astype(np.uint8))
        plt.title(f'{model_name} Channel {channel_idx} Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_{layer_name}_top_channels.png', dpi=300)
    plt.show()

def overlay_activations_on_image(image, feature_map, layer_name, model_name):
    """Show overall activation overlay across all channels"""
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
    plt.title(f'{model_name} {layer_name} Activation Overlay')
    plt.axis('off')
    
    plt.savefig(f'{model_name}_{layer_name}_overlay.png', dpi=300)
    plt.show()

def compare_resnet_models(image, frame_idx=150):
    """Compare ResNet18 and ResNet34 on the same image"""
    print(f"\n===== Comparing ResNet18 vs ResNet34 on frame {frame_idx} =====")
    
    # Preprocess the image once
    image_tensor = preprocess_image(image)
    
    # Create side-by-side comparison table
    comparison_data = []
    
    # Process with ResNet18
    print("Processing with ResNet18...")
    model18, layers18 = get_resnet_model('resnet18')
    model18.eval()
    features18 = extract_features(model18, layers18, image_tensor)
    
    # Get layer4 parameters
    layer4_params_18 = sum(p.numel() for p in model18.layer4.parameters())
    comparison_data.append(("ResNet18", layer4_params_18, features18['layer4'].shape))
    
    # Process with ResNet34
    print("Processing with ResNet34...")
    model34, layers34 = get_resnet_model('resnet34')
    model34.eval()
    features34 = extract_features(model34, layers34, image_tensor)
    
    # Get layer4 parameters
    layer4_params_34 = sum(p.numel() for p in model34.layer4.parameters())
    comparison_data.append(("ResNet34", layer4_params_34, features34['layer4'].shape))
    
    # Print comparison table
    print("\nModel Comparison:")
    print("-" * 70)
    print(f"{'Model':<10} | {'Layer4 Parameters':<20} | {'Layer4 Output Shape':<30}")
    print("-" * 70)
    for model_name, params, shape in comparison_data:
        print(f"{model_name:<10} | {params:<20,} | {str(shape):<30}")
    print("-" * 70)
    
    # Visualize ResNet18 results
    print("\nVisualizing ResNet18 layer4 activations...")
    overlay_activations_on_image(image, features18['layer4'], 'layer4', 'ResNet18')
    visualize_top_channels(image, features18['layer4'], 'layer4', 'ResNet18')
    
    # Visualize ResNet34 results
    print("\nVisualizing ResNet34 layer4 activations...")
    overlay_activations_on_image(image, features34['layer4'], 'layer4', 'ResNet34')
    visualize_top_channels(image, features34['layer4'], 'layer4', 'ResNet34')
    
    # Return the features for further analysis if needed
    return features18, features34

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
    
    # Load frames from different parts of the episode
    frames = [30, 150, 270]  # Beginning, middle, end of episode
    
    for frame_idx in frames:
        # Load image
        image = load_image_from_hdf5(dataset_path, frame_idx=frame_idx)
        print(f"Processing frame {frame_idx}, Image shape: {image.shape}")
        
        # Compare models
        compare_resnet_models(image, frame_idx)

if __name__ == "__main__":
    main()