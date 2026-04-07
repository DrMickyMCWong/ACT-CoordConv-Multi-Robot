#!/usr/bin/env python3
"""
Convert raw_episodes to task7c format.

This script converts episodes from the new recording format to the format
expected by the ACT training pipeline.

Changes made:
1. File naming: raw_episode_XXXX_timestamp.h5 → episode_X.hdf5
2. Structure: Flat HDF5 → Nested (observations group)
3. Image resize: 720x1280 → 480x640
4. Key mapping: state → qpos, color_image → images/front
5. Add qvel: Create zeros array
6. Remove: point_cloud and timestamps
7. Add 'sim' attribute: Set to False (real robot data)
"""

import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def resize_image(image, target_height=480, target_width=640):
    """Resize image from 720x1280 to 480x640."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def convert_episode(input_path, output_path, episode_num):
    """Convert a single episode from raw format to task7c format."""
    
    with h5py.File(input_path, 'r') as src:
        # Read data from source
        action = src['action'][()]
        state = src['state'][()]  # This will become qpos
        color_image = src['color_image'][()]
        # point_cloud and timestamps are ignored
        
        episode_len = action.shape[0]
        
        # Create qvel as zeros (velocity not recorded, but required by training)
        qvel = np.zeros_like(state)
        
        # Resize images
        resized_images = np.zeros((episode_len, 480, 640, 3), dtype=np.uint8)
        print(f"  Resizing {episode_len} images from 720x1280 to 480x640...")
        for i in range(episode_len):
            resized_images[i] = resize_image(color_image[i])
        
    # Write to new file
    with h5py.File(output_path, 'w') as dst:
        # Add sim attribute (False for real robot data)
        dst.attrs['sim'] = False
        
        # Create action dataset (unchanged)
        dst.create_dataset('action', data=action)
        
        # Create observations group
        obs_group = dst.create_group('observations')
        
        # Add qpos and qvel
        obs_group.create_dataset('qpos', data=state)
        obs_group.create_dataset('qvel', data=qvel)
        
        # Create images group and add front camera
        images_group = obs_group.create_group('images')
        images_group.create_dataset('front', data=resized_images)
    
    print(f"  ✓ Converted episode_{episode_num}.hdf5 (length: {episode_len})")

def main():
    parser = argparse.ArgumentParser(description='Convert raw episodes to task7c format')
    parser.add_argument('--input_dir', type=str, 
                        default='/home/hk/Documents/ACT_Shaka/data/raw_episodes',
                        help='Directory containing raw episode files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/hk/Documents/ACT_Shaka/data/task7c_new',
                        help='Directory to save converted episodes')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting episode index (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending episode index (default: all)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of raw episode files
    raw_files = sorted([f for f in os.listdir(args.input_dir) if f.startswith('raw_episode_') and f.endswith('.h5')])
    
    if args.end_idx is not None:
        raw_files = raw_files[args.start_idx:args.end_idx]
    else:
        raw_files = raw_files[args.start_idx:]
    
    print(f"Found {len(raw_files)} episodes to convert")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Convert each episode
    for idx, raw_file in enumerate(tqdm(raw_files, desc="Converting episodes")):
        episode_num = args.start_idx + idx
        input_path = os.path.join(args.input_dir, raw_file)
        output_path = os.path.join(args.output_dir, f'episode_{episode_num}.hdf5')
        
        try:
            convert_episode(input_path, output_path, episode_num)
        except Exception as e:
            print(f"  ✗ Error converting {raw_file}: {e}")
            continue
    
    print("="*60)
    print(f"✓ Conversion complete! {len(raw_files)} episodes converted.")
    print(f"\nConverted episodes saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Verify a few converted episodes")
    print("2. Update config to point to the new dataset directory")
    print("3. Run training with the converted dataset")

if __name__ == '__main__':
    main()
