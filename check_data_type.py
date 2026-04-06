import h5py
import numpy as np
import os

def check_data_types(file_path, label):
    """Check data types in HDF5 file"""
    print(f"\n=== {label} ===")
    print(f"File: {file_path}")
    
    with h5py.File(file_path, 'r') as root:
        # Check qpos dtype
        qpos_dtype = root['/observations/qpos'].dtype
        qpos_shape = root['/observations/qpos'].shape
        print(f"QPos: dtype={qpos_dtype}, shape={qpos_shape}")
        
        # Check qvel dtype  
        qvel_dtype = root['/observations/qvel'].dtype
        qvel_shape = root['/observations/qvel'].shape
        print(f"QVel: dtype={qvel_dtype}, shape={qvel_shape}")
        
        # Check action dtype
        action_dtype = root['/action'].dtype
        action_shape = root['/action'].shape
        print(f"Action: dtype={action_dtype}, shape={action_shape}")
        
        # Check image dtype
        for cam_name in root['/observations/images'].keys():
            img_dtype = root[f'/observations/images/{cam_name}'].dtype
            img_shape = root[f'/observations/images/{cam_name}'].shape
            print(f"Image {cam_name}: dtype={img_dtype}, shape={img_shape}")
        
        # Sample some values
        print(f"Sample QPos values: {root['/observations/qpos'][0]}")
        print(f"Sample Action values: {root['/action'][0]}")

# Check both files
original_file = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task2\episode_0.hdf5'
corrected_file = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task2c\episode_0.hdf5'

check_data_types(original_file, "ORIGINAL FILE")
if os.path.exists(corrected_file):
    check_data_types(corrected_file, "CORRECTED FILE")
else:
    print(f"Corrected file not found: {corrected_file}")