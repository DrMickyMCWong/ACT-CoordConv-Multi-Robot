from robot import Robot
from config.config import ROBOT_PORTS
import time
import sys

def main():
    print("Starting teleoperation...")
    
    # init robots with error handling
    try:
        leader = Robot(device_name=ROBOT_PORTS['leader'])
        follower = Robot(device_name=ROBOT_PORTS['follower'])
    except Exception as e:
        print(f"Error initializing robots: {e}")
        sys.exit(1)
    
    # Give time for serial connections to stabilize
    time.sleep(2)
    
    print("Activating leader arm...")
    # activate the leader gripper torque and configure ESP-NOW
    leader._disable_torque()
    time.sleep(1)  # Wait for ESP-NOW pairing to establish
    
    print("Teleoperation active! Press Ctrl+C to exit.")
    try:
        # Keep track of successful reads
        successful_reads = 0
        failed_reads = 0
        
        while True:
            # Try to read position from leader
            try:
                position = leader.read_position()
                
                # Check if we got valid data (non-zero values)
                if any(position):
                    successful_reads += 1
                    print(f"\rSuccessful reads: {successful_reads}, Failed reads: {failed_reads}", end="")
                else:
                    # Got zeros, likely a read error
                    failed_reads += 1
                    print(f"\rSuccessful reads: {successful_reads}, Failed reads: {failed_reads}", end="")
                    time.sleep(0.1)  # Extra delay on failure
                    continue
                    
                # Position is automatically sent via ESP-NOW
                # No need to call set_goal_pos as this happens via ESP32 communication
                
                # Short delay to prevent overwhelming the serial port
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed_reads += 1
                print(f"\nError during teleoperation: {e}")
                time.sleep(0.1)  # Delay to recover from errors
    
    except KeyboardInterrupt:
        print("\n\nTeleoperation stopped by user.")
    finally:
        # Clean up and disconnect
        print("Disabling torque and cleaning up...")
        leader._disable_torque()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()