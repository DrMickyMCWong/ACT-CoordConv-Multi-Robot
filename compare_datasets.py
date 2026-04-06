import h5py
import numpy as np

print("="*60)
print("EXAMINING NEW RAW EPISODE")
print("="*60)
with h5py.File('/home/hk/Documents/ACT_Shaka/data/raw_episodes/raw_episode_0000_20251121_192511.h5', 'r') as f:
    print(f"Keys: {list(f.keys())}\n")
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"{key}:")
            print(f"  Shape: {f[key].shape}")
            print(f"  Dtype: {f[key].dtype}")
            print(f"  Sample data: {f[key][0]}")
            print()

print("\n" + "="*60)
print("EXAMINING TASK7C EPISODE")
print("="*60)
with h5py.File('/home/hk/Documents/ACT_Shaka/data/task7c/episode_0.hdf5', 'r') as f:
    print(f"Keys: {list(f.keys())}\n")
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}:")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
            if len(obj.shape) > 0:
                print(f"  Sample data: {obj[0] if obj.shape[0] > 0 else 'empty'}")
            print()
    
    f.visititems(print_structure)

print("\n" + "="*60)
print("SUMMARY OF DIFFERENCES")
print("="*60)
