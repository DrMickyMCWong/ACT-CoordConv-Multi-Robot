import h5py
import numpy as np

def verify_episode(filepath):
    """Verify an episode file has the correct structure."""
    print(f"Verifying: {filepath}")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        # Check attributes
        print(f"Attributes:")
        print(f"  sim: {f.attrs['sim']}")
        print()
        
        # Check structure
        print(f"Root keys: {list(f.keys())}")
        print()
        
        # Check action
        action = f['action']
        print(f"action:")
        print(f"  Shape: {action.shape}")
        print(f"  Dtype: {action.dtype}")
        print()
        
        # Check observations
        print(f"observations keys: {list(f['observations'].keys())}")
        print()
        
        # Check qpos
        qpos = f['observations/qpos']
        print(f"observations/qpos:")
        print(f"  Shape: {qpos.shape}")
        print(f"  Dtype: {qpos.dtype}")
        print()
        
        # Check qvel
        qvel = f['observations/qvel']
        print(f"observations/qvel:")
        print(f"  Shape: {qvel.shape}")
        print(f"  Dtype: {qvel.dtype}")
        print(f"  All zeros: {np.allclose(qvel[()], 0)}")
        print()
        
        # Check images
        print(f"observations/images keys: {list(f['observations/images'].keys())}")
        print()
        
        front = f['observations/images/front']
        print(f"observations/images/front:")
        print(f"  Shape: {front.shape}")
        print(f"  Dtype: {front.dtype}")
        print()
        
        # Verify shapes match
        assert action.shape[0] == qpos.shape[0] == qvel.shape[0] == front.shape[0], \
            "Episode length mismatch!"
        print(f"✓ All datasets have consistent episode length: {action.shape[0]}")
        print(f"✓ Image resolution: {front.shape[1]}x{front.shape[2]} (expected 480x640)")
        print(f"✓ State dimensions: {qpos.shape[1]} (expected 6)")
        print(f"✓ Action dimensions: {action.shape[1]} (expected 6)")
        print()

print("CONVERTED EPISODE (NEW)")
print("="*60)
verify_episode('/home/hk/Documents/ACT_Shaka/data/task7c_new/episode_0.hdf5')

print("\n\n")
print("ORIGINAL TASK7C EPISODE (REFERENCE)")
print("="*60)
verify_episode('/home/hk/Documents/ACT_Shaka/data/task7c/episode_0.hdf5')
