#!/usr/bin/env python3
"""
Quick test script to verify robot arm connection on Linux
"""

import os
import sys

# Check available USB devices
print("="*60)
print("Checking available USB serial devices...")
print("="*60)

# List all ttyUSB devices
os.system("ls -l /dev/ttyUSB* 2>/dev/null || echo 'No /dev/ttyUSB* devices found'")

print("\n" + "="*60)
print("Checking device permissions...")
print("="*60)

# Check if user has permission to access serial ports
import subprocess
result = subprocess.run(['groups'], capture_output=True, text=True)
groups = result.stdout.strip()
print(f"Your user groups: {groups}")

if 'dialout' not in groups:
    print("\n⚠️  WARNING: You are not in the 'dialout' group!")
    print("To access serial ports, run:")
    print(f"  sudo usermod -a -G dialout $USER")
    print("Then log out and log back in.")
else:
    print("✅ You are in the 'dialout' group - serial access should work")

print("\n" + "="*60)
print("Testing robot connection...")
print("="*60)

try:
    from config.config import ROBOT_PORTS
    from robot import Robot
    
    print(f"\nAttempting to connect to follower arm at: {ROBOT_PORTS['follower']}")
    
    # Try to initialize the robot
    follower = Robot(device_name=ROBOT_PORTS['follower'])
    
    print("✅ Robot connection successful!")
    print(f"   Port: {ROBOT_PORTS['follower']}")
    
    # Try to read position
    print("\nTesting position read...")
    pos = follower.read_position()
    print(f"✅ Current position: {pos}")
    
    print("\n" + "="*60)
    print("🎉 All tests passed! Ready for evaluation.")
    print("="*60)
    
except FileNotFoundError as e:
    print(f"❌ Error: Device not found - {e}")
    print("\nTroubleshooting:")
    print("1. Check if the robot arm is connected via USB")
    print("2. Run: ls -l /dev/ttyUSB*")
    print("3. Verify the correct port in config/config.py")
    
except PermissionError as e:
    print(f"❌ Error: Permission denied - {e}")
    print("\nTo fix:")
    print("  sudo usermod -a -G dialout $USER")
    print("  Then log out and log back in")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease check:")
    print("1. Robot arm is powered on")
    print("2. USB cable is connected")
    print("3. Correct port specified in config/config.py")
