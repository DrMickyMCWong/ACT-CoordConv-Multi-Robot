import h5py
import os

def check_camera_names(file_path):
    """Check what camera names exist in an HDF5 file"""
    print(f"Checking file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File does not exist!")
        return []
    
    with h5py.File(file_path, 'r') as root:
        print(f"Root keys: {list(root.keys())}")
        
        if '/observations' in root:
            print(f"Observations keys: {list(root['/observations'].keys())}")
            
            if '/observations/images' in root:
                camera_names = list(root['/observations/images'].keys())
                print(f"Camera names found: {camera_names}")
                
                # Check each camera's data
                for cam in camera_names:
                    shape = root[f'/observations/images/{cam}'].shape
                    dtype = root[f'/observations/images/{cam}'].dtype
                    print(f"  {cam}: shape={shape}, dtype={dtype}")
                
                return camera_names
            else:
                print("No /observations/images found!")
        else:
            print("No /observations found!")
    
    return []

# Check your original dataset
original_file = r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\data\task2\episode_0.hdf5'
print("=== ORIGINAL DATASET ===")
orig_cameras = check_camera_names(original_file)

# Check training config camera names
print("\n=== CHECKING CONFIG ===")
try:
    from config.config import TASK_CONFIG
    print(f"Config camera names: {TASK_CONFIG.get('camera_names', 'Not found')}")
except:
    print("Could not import config")