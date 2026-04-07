import os

def find_camera_references():
    """Find camera name references in training files"""
    
    # Files to check
    files_to_check = [
        r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\training\utils.py',
        r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\train.py',
        r'C:\Users\Administrator\Documents\transformer\ACT-Shaka\config\config.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n=== CHECKING {file_path} ===")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                # Look for camera-related lines
                if any(keyword in line.lower() for keyword in ['camera', 'front', 'back', 'left', 'right', 'top', 'cam']):
                    print(f"Line {i}: {line.strip()}")
        else:
            print(f"File not found: {file_path}")

find_camera_references()