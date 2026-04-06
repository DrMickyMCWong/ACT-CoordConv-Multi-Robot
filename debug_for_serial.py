import numpy as np
import pandas as pd
import time
import math
import os
import h5py
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        action = root['/action'][()]

    return qpos, qvel, action, image_dict

# Load into df
data_file = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task1\episode_0.hdf5'
qpos, qvel, action, image_dict = load_hdf5(dataset_path=data_file)

# Load qpos and action into DataFrames with joint headers
joint_headers = ['timestamp', 'b', 's', 'e', 't', 'r', 'g']

# Create timestamp column (assuming sequential timesteps)
num_timesteps = len(qpos)
timestamps = np.arange(num_timesteps)

# Create qpos DataFrame
qpos_data = np.column_stack([timestamps, qpos])
qpos_df = pd.DataFrame(qpos_data, columns=joint_headers)

# Create action DataFrame
action_data = np.column_stack([timestamps, action])
action_df = pd.DataFrame(action_data, columns=joint_headers)

print("QPos DataFrame shape:", qpos_df.shape)
print("Action DataFrame shape:", action_df.shape)
print("\nQPos DataFrame head:")
print(qpos_df.head())
print("\nAction DataFrame head:")
print(action_df.head())

# Optional: Check the data ranges for each joint
print("\nQPos data ranges:")
for col in joint_headers[1:]:  # Skip timestamp
    print(f"{col}: {qpos_df[col].min():.3f} to {qpos_df[col].max():.3f}")

print("\nAction data ranges:")
for col in joint_headers[1:]:  # Skip timestamp
    print(f"{col}: {action_df[col].min():.3f} to {action_df[col].max():.3f}")
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fix_temporal_alignment_with_first_qpos(qpos_df, action_df, joint_headers):
    """
    Fix temporal alignment by:
    1. Copy first row of QPos and insert it at the beginning (duplicate first position)
    2. Copy QPos[t+1] to Action[t] (actions should predict next states) 
    3. Truncate to maintain same length
    
    Result: QPos becomes [qpos[0], qpos[0], qpos[1], qpos[2], ...]
            Action becomes [qpos[0], qpos[1], qpos[2], qpos[3], ...]
    """
    
    joint_cols = ['b', 's', 'e', 't', 'r', 'g']
    
    # Create corrected dataframes
    qpos_corrected = qpos_df.copy()
    action_corrected = action_df.copy()
    
    # Step 1: Copy first row of QPos and insert at beginning
    first_qpos_row = qpos_df.iloc[0].copy()
    first_qpos_row['timestamp'] = 0  # Set timestamp to 0 for the duplicated row
    
    # Insert the first row at the beginning
    qpos_corrected = pd.concat([pd.DataFrame([first_qpos_row]), qpos_corrected], ignore_index=True)
    
    # Step 2: Copy QPos[t+1] to Action[t] (actions predict next states)
    # Now qpos_corrected has: [qpos[0], qpos[0], qpos[1], qpos[2], ...]
    # We want action to be:     [qpos[0], qpos[1], qpos[2], qpos[3], ...]
    for joint in joint_cols:
        # Use the shifted QPos as action targets
        action_corrected[joint] = qpos_corrected[joint][1:len(action_corrected)+1].values
    
    # Step 3: Truncate QPos to maintain same length as action
    qpos_corrected = qpos_corrected[:-1].reset_index(drop=True)
    
    # Update timestamps to be sequential
    qpos_corrected['timestamp'] = range(len(qpos_corrected))
    action_corrected['timestamp'] = range(len(action_corrected))
    
    return qpos_corrected, action_corrected

# Apply the fix
print("Applying temporal alignment fix using first QPos row...")
qpos_fixed, action_fixed = fix_temporal_alignment_with_first_qpos(qpos_df, action_df, joint_headers)

print(f"Original QPos shape: {qpos_df.shape}")
print(f"Original Action shape: {action_df.shape}")
print(f"Fixed QPos shape: {qpos_fixed.shape}")
print(f"Fixed Action shape: {action_fixed.shape}")

print("\nInitial positions check:")
print("Original QPos first row:")
print(qpos_df.iloc[0])
print("\nFixed QPos first row (should be copy of original first row):")
print(qpos_fixed.iloc[0])
print("\nFixed QPos second row (should be same as first row):")
print(qpos_fixed.iloc[1])
print("\nFixed Action first row (should match original QPos first row):")
print(action_fixed.iloc[0])
print("\nFixed Action second row (should match original QPos second row):")
print(action_fixed.iloc[1])

# Verify the mapping
print("\nVerification of QPos->Action mapping:")
print("Format: QPos[t] -> Action[t] (should predict QPos[t+1])")
print("-" * 60)
joint_cols = ['b', 's', 'e', 't', 'r', 'g']
for i in range(min(5, len(qpos_fixed)-1)):
    print(f"Timestep {i}:")
    for joint in joint_cols[:3]:  # Show first 3 joints for brevity
        qpos_t = qpos_fixed[joint].iloc[i]
        action_t = action_fixed[joint].iloc[i]
        qpos_t_plus_1 = qpos_fixed[joint].iloc[i+1]
        match = "✓" if abs(action_t - qpos_t_plus_1) < 1e-6 else "✗"
        print(f"  {joint}: QPos[{i}]={qpos_t:.3f} -> Action[{i}]={action_t:.3f} (should equal QPos[{i+1}]={qpos_t_plus_1:.3f}) {match}")
    print()
    
def save_corrected_hdf5(qpos_fixed, action_fixed, qvel, image_dict, output_path, original_attrs=None):
    """
    Save the corrected qpos, action data along with unchanged qvel and image_dict 
    back to HDF5 format matching the original structure
    """
    
    joint_cols = ['b', 's', 'e', 't', 'r', 'g']
    
    # Convert DataFrames back to numpy arrays (excluding timestamp column)
    qpos_corrected_array = qpos_fixed[joint_cols].values
    action_corrected_array = action_fixed[joint_cols].values
    
    # Ensure qvel matches the length of corrected data
    qvel_corrected = qvel[:len(qpos_corrected_array)]
    
    # Ensure image data matches the length of corrected data
    image_dict_corrected = {}
    for cam_name, images in image_dict.items():
        image_dict_corrected[cam_name] = images[:len(qpos_corrected_array)]
    
    print(f"Saving corrected data:")
    print(f"  QPos shape: {qpos_corrected_array.shape}")
    print(f"  Action shape: {action_corrected_array.shape}")
    print(f"  QVel shape: {qvel_corrected.shape}")
    for cam_name, images in image_dict_corrected.items():
        print(f"  {cam_name} images shape: {images.shape}")
    
    # Create the HDF5 file with the same structure as original
    with h5py.File(output_path, 'w') as f:
        if original_attrs is not None:
            for key, value in original_attrs.items():
                f.attrs[key] = value
        else:
            # Set default attributes if not provided
            f.attrs['sim'] = False  # or False, depending on your data

        # Create observations group
        obs_group = f.create_group('observations')
        
        # Save qpos and qvel
        obs_group.create_dataset('qpos', data=qpos_corrected_array, dtype=np.float32)
        obs_group.create_dataset('qvel', data=qvel_corrected, dtype=np.float32)

        # Create images group and save all camera data
        images_group = obs_group.create_group('images')
        for cam_name, images in image_dict_corrected.items():
            images_group.create_dataset(cam_name, data=images)
        
        # Save corrected actions
        f.create_dataset('action', data=action_corrected_array, dtype=np.float32)

    print(f"✓ Corrected data saved to: {output_path}")
    
    # Verify the saved file
    print("\nVerification - loading saved file:")
    qpos_verify, qvel_verify, action_verify, image_dict_verify = load_hdf5(output_path)
    print(f"  Loaded QPos shape: {qpos_verify.shape}")
    print(f"  Loaded Action shape: {action_verify.shape}")
    print(f"  Loaded QVel shape: {qvel_verify.shape}")
    for cam_name, images in image_dict_verify.items():
        print(f"  Loaded {cam_name} images shape: {images.shape}")

def process_all_episodes(input_folder, output_folder):
    """
    Process all HDF5 files in input_folder and save corrected versions to output_folder
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all HDF5 files in input folder
    input_pattern = os.path.join(input_folder, "*.hdf5")
    hdf5_files = glob.glob(input_pattern)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_folder}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process:")
    for file in hdf5_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    for i, input_file in enumerate(hdf5_files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(hdf5_files)}: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        
        try:
            # Read original attributes first
            original_attrs = {}
            with h5py.File(input_file, 'r') as f:
                for key in f.attrs.keys():
                    original_attrs[key] = f.attrs[key]
            
            # Load the original data
            qpos, qvel, action, image_dict = load_hdf5(input_file)
            
            # Convert to DataFrames
            joint_headers = ['timestamp', 'b', 's', 'e', 't', 'r', 'g']
            num_timesteps = len(qpos)
            timestamps = np.arange(num_timesteps)
            
            qpos_data = np.column_stack([timestamps, qpos])
            qpos_df = pd.DataFrame(qpos_data, columns=joint_headers)
            
            action_data = np.column_stack([timestamps, action])
            action_df = pd.DataFrame(action_data, columns=joint_headers)
            
            print(f"Original data shapes: QPos {qpos_df.shape}, Action {action_df.shape}")
            
            # Apply temporal alignment fix
            qpos_fixed, action_fixed = fix_temporal_alignment_with_first_qpos(qpos_df, action_df, joint_headers)
            
            print(f"Fixed data shapes: QPos {qpos_fixed.shape}, Action {action_fixed.shape}")
            
            # Create output file path (same filename, different folder)
            input_filename = os.path.basename(input_file)
            output_file = os.path.join(output_folder, input_filename)
            
            # Save the corrected data
            save_corrected_hdf5(qpos_fixed, action_fixed, qvel, image_dict, output_file)
            
            print(f"✓ Successfully processed and saved: {input_filename}")
            
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(input_file)}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Summary
    output_files = glob.glob(os.path.join(output_folder, "*.hdf5"))
    print(f"Successfully processed: {len(output_files)}/{len(hdf5_files)} files")
    
    if output_files:
        print("\nCreated files:")
        for file in sorted(output_files):
            print(f"  - {os.path.basename(file)}")

# Define input and output folders
input_folder = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task7'
output_folder = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task7c'

# Process all episodes
process_all_episodes(input_folder, output_folder)