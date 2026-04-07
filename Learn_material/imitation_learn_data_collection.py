"""
Simplified Trajectory Data Collection System
Clean data collection with CSV output and reset functionality
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from openni import openni2
import time
import os
import json
import pandas as pd
from datetime import datetime
import threading
import queue
import serial
import serial.tools.list_ports
from collections import deque

class SimplifiedTrajectoryDataCollector:
    def __init__(self, yolo_model_path, serial_port='COM5'):
        """Initialize simplified trajectory data collection system"""
        self.yolo_model_path = yolo_model_path
        self.serial_port = serial_port
        
        # Load YOLO model
        print(f"📦 Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize cameras
        self.setup_cameras()
        
        # Initialize serial communication with arm
        self.setup_serial()
        
        # Data collection state
        self.recording = False
        self.current_trajectory = []
        self.csv_data = []  # Changed from trajectory_data to csv_data
        self.sample_count = 0
        self.rest_position = None
        self.target_brick = None
        
        # Simplified joint data management (removed load and additional params)
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.data_lock = threading.Lock()
        
        # Data collection settings
        self.recording_frequency = 10  # Hz
        self.min_trajectory_length = 5
        
        # Threading control
        self.running = True
        
        # Output file - CSV only
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"trajectory_data_{timestamp}.csv"
        
        # Start background thread
        self.start_background_thread()
        
        print("✅ Simplified Trajectory Data Collector initialized")
        print(f"📁 Output file: {self.output_file}")
    
    def setup_cameras(self):
        """Initialize RGB and depth cameras"""
        print("📷 Setting up cameras...")
        try:
            # Initialize OpenNI for depth
            openni2.initialize()
            self.dev = openni2.Device.open_any()
            self.depth_stream = self.dev.create_depth_stream()
            self.depth_stream.start()
            
            # Initialize RGB camera
            self.rgb_cap = cv2.VideoCapture(0)
            self.rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            time.sleep(2)
            print("✅ Cameras initialized")
            
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            exit(1)
    
    def setup_serial(self):
        """Initialize serial communication with arm"""
        print(f"🔌 Setting up serial communication on {self.serial_port}...")
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=115200,
                timeout=0.1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)
            print("✅ Serial communication initialized")
            
        except Exception as e:
            print(f"❌ Serial setup failed: {e}")
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            print(f"🔍 Available ports: {available_ports}")
            self.ser = None
    
    def start_background_thread(self):
        """Start background thread for continuous data reading"""
        self.arm_thread = threading.Thread(target=self.continuous_arm_reading, daemon=True)
        self.arm_thread.start()
        print("✅ Background arm data reading thread started")
    
    def continuous_arm_reading(self):
        """Continuously read arm position in background thread"""
        while self.running:
            arm_data = self.get_arm_data_immediate()
            if arm_data:
                with self.data_lock:
                    self.current_joints = [
                        arm_data['joints']['base'],
                        arm_data['joints']['shoulder'],
                        arm_data['joints']['elbow'],
                        arm_data['joints']['turn'],
                        arm_data['joints']['roll']
                    ]
            time.sleep(0.01)  # 100Hz reading rate
    
    def get_arm_data_immediate(self):
        """Read current arm data immediately from serial"""
        if not self.ser:
            return None
        
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                
                if line:
                    arm_data = json.loads(line)
                    
                    # Extract joint angles (radians) - only joint angles needed
                    base = arm_data.get('b', None)
                    shoulder = arm_data.get('s', None)
                    elbow = arm_data.get('e', None)
                    turn = arm_data.get('t', None)
                    roll = arm_data.get('r', None)
                    
                    # Validate data
                    joints_valid = all(v is not None for v in [base, shoulder, elbow, turn, roll])
                    
                    if joints_valid:
                        return {
                            'joints': {
                                'base': base,
                                'shoulder': shoulder,
                                'elbow': elbow,
                                'turn': turn,
                                'roll': roll
                            }
                        }
                        
        except (json.JSONDecodeError, UnicodeDecodeError, serial.SerialException) as e:
            pass
        
        return None
    
    def get_depth_at_point(self, u, v):
        """Get depth value at pixel coordinates"""
        try:
            for _ in range(2):
                depth_frame = self.depth_stream.read_frame()
            
            if depth_frame is not None:
                depth_data = np.frombuffer(
                    depth_frame.get_buffer_as_uint16(),
                    dtype=np.uint16
                ).reshape(480, 640)
                
                depth_data = np.fliplr(depth_data)
                
                if 0 <= v < 480 and 0 <= u < 640:
                    depth_value = depth_data[v, u]
                    
                    if depth_value == 0:
                        u_min, u_max = max(0, u-1), min(640, u+2)
                        v_min, v_max = max(0, v-1), min(480, v+2)
                        depth_region = depth_data[v_min:v_max, u_min:u_max]
                        valid_depths = depth_region[depth_region > 0]
                        if len(valid_depths) > 0:
                            depth_value = int(np.median(valid_depths))
                    
                    return depth_value if depth_value > 0 else None
        except Exception as e:
            print(f"⚠️ Error getting depth: {e}")
        return None
    
    def detect_brick(self, image):
        """Detect brick and return its position"""
        results = self.yolo_model(image, verbose=False)
        
        best_brick = None
        best_confidence = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    class_name = self.yolo_model.names[cls]
                    
                    if class_name.lower() == 'brick':
                        conf = box.conf[0].item()
                        if conf > best_confidence:
                            x_center, y_center, width, height = box.xywh[0].tolist()
                            u, v = int(x_center), int(y_center)
                            depth_value = self.get_depth_at_point(u, v)
                            
                            if depth_value and depth_value > 0:
                                best_brick = {
                                    'u': u,
                                    'v': v,
                                    'depth': int(depth_value),
                                    'confidence': conf
                                }
                                best_confidence = conf
        
        return best_brick
    
    def get_current_joint_angles(self):
        """Get current joint angles"""
        with self.data_lock:
            return self.current_joints.copy()
    
    def get_current_state(self):
        """Get current complete state - simplified"""
        with self.data_lock:
            return {
                'joint_angles': [float(j) for j in self.current_joints]
            }
    
    def set_rest_position(self):
        """Set the current position as rest position"""
        self.rest_position = self.get_current_joint_angles()
        print(f"✅ Rest position set: {[f'{j:.3f}' for j in self.rest_position]}")
    
    def reset_system(self):
        """Reset the entire system to initial state"""
        print("🔄 RESETTING SYSTEM...")
        
        # Stop any current recording
        self.recording = False
        
        # Clear all data
        self.current_trajectory = []
        self.rest_position = None
        self.target_brick = None
        
        print("✅ System reset complete!")
        print("📋 Status:")
        print("   - Recording: STOPPED")
        print("   - Rest position: CLEARED")
        print("   - Current trajectory: CLEARED")
        print("   - Target brick: CLEARED")
        print("👉 Ready for new recording session")
    
    def start_recording(self, brick_info):
        """Start recording trajectory"""
        if self.rest_position is None:
            print("❌ Please set rest position first (press 'R')")
            return
        
        if not self.ser:
            print("❌ No serial connection to arm")
            return
        
        self.recording = True
        self.current_trajectory = []
        self.target_brick = brick_info
        
        # Record initial state
        initial_state = self.get_current_state()
        initial_state['timestamp'] = time.time()
        initial_state['step'] = 0
        
        self.current_trajectory.append(initial_state)
        
        print(f"🎬 RECORDING STARTED - Sample {self.sample_count + 1}")
        print(f"🎯 Target brick: ({brick_info['u']}, {brick_info['v']}, {brick_info['depth']}mm)")
        print("👋 Move arm to brick position, then press SPACEBAR to stop recording")

    def stop_recording(self):
        """Stop recording trajectory"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Record final state
        final_state = self.get_current_state()
        final_state['timestamp'] = time.time()
        final_state['step'] = len(self.current_trajectory)
        
        self.current_trajectory.append(final_state)
        
        if len(self.current_trajectory) >= self.min_trajectory_length:
            print(f"✅ RECORDING STOPPED - Trajectory captured!")
            print(f"📊 Duration: {final_state['timestamp'] - self.current_trajectory[0]['timestamp']:.2f}s")
            print(f"📊 Steps: {len(self.current_trajectory)}")
            print("🔄 Move arm back to REST position, then press 'N' to finish episode")
        else:
            print(f"❌ Trajectory too short ({len(self.current_trajectory)} steps), discarded")
            self.current_trajectory = []

    def finish_episode(self):
        """Finish current episode and save trajectory"""
        if not self.current_trajectory:
            print("❌ No trajectory to save")
            return
        
        if not self.target_brick:
            print("❌ No target brick information")
            return
        
        # Convert trajectory to CSV format immediately
        sample_id = self.sample_count
        brick_info = self.target_brick
        rest_pos = self.rest_position
        
        for step_data in self.current_trajectory:
            joint_angles = step_data['joint_angles']
            
            row = {
                'sample_id': sample_id,
                'step': step_data['step'],
                'timestamp': step_data['timestamp'],
                'brick_u': brick_info['u'],
                'brick_v': brick_info['v'],
                'brick_depth': brick_info['depth'],
                'brick_confidence': brick_info['confidence'],
                'rest_base': rest_pos[0],
                'rest_shoulder': rest_pos[1],
                'rest_elbow': rest_pos[2],
                'rest_turn': rest_pos[3],
                'rest_roll': rest_pos[4],
                'joint_base': joint_angles[0],
                'joint_shoulder': joint_angles[1],
                'joint_elbow': joint_angles[2],
                'joint_turn': joint_angles[3],
                'joint_roll': joint_angles[4]
            }
            self.csv_data.append(row)
        
        self.sample_count += 1
        
        print(f"✅ EPISODE FINISHED - Trajectory saved to CSV!")
        print(f"📈 Total samples collected: {self.sample_count}")
        print(f"📊 Total data points: {len(self.csv_data)}")
        
        # Clear current trajectory and reset rest position
        self.current_trajectory = []
        self.rest_position = None
        self.target_brick = None
        
        # Auto-save CSV
        self.save_csv_data()

    def record_trajectory_step(self):
        """Record a single trajectory step"""
        if self.recording:
            step_data = self.get_current_state()
            step_data['timestamp'] = time.time()
            step_data['step'] = len(self.current_trajectory)
            
            self.current_trajectory.append(step_data)

    def save_csv_data(self):
        """Save trajectory data to CSV file"""
        if not self.csv_data:
            print("❌ No data to save")
            return
        
        try:
            df = pd.DataFrame(self.csv_data)
            df.to_csv(self.output_file, index=False)
            print(f"💾 CSV data saved to {self.output_file}")
            print(f"📈 Total rows: {len(df)}")
        except Exception as e:
            print(f"❌ Error saving CSV data: {e}")
    
    def create_display_frame(self, image, brick_info=None):
        """Create display frame with UI"""
        img_copy = image.copy()
        
        # Instructions
        cv2.putText(img_copy, "TRAJECTORY DATA COLLECTION - CSV OUTPUT", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_copy, "R: Set Rest Position", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_copy, "SPACEBAR: Start/Stop Recording", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_copy, "N: Finish Episode", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_copy, "C: RESET System", (10, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(img_copy, "S: Save CSV Data", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_copy, "ESC: Exit", (10, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status
        status_y = 215
        if self.rest_position:
            cv2.putText(img_copy, f"Rest Position: SET", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img_copy, f"Rest Position: NOT SET", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(img_copy, f"Episodes: {self.sample_count}", (10, status_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(img_copy, f"CSV Rows: {len(self.csv_data)}", (10, status_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Serial status
        serial_status = "Connected" if self.ser else "Disconnected"
        serial_color = (0, 255, 0) if self.ser else (0, 0, 255)
        cv2.putText(img_copy, f"Serial: {serial_status}", (10, status_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, serial_color, 2)
        
        # Recording status
        if self.recording:
            cv2.putText(img_copy, f"RECORDING... Steps: {len(self.current_trajectory)}", (10, status_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.current_trajectory:
            cv2.putText(img_copy, f"EPISODE READY - Press 'N' to finish", (10, status_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Current joint angles
        current_joints = self.get_current_joint_angles()
        joint_text = f"Joints: {[f'{j:.3f}' for j in current_joints]}"
        cv2.putText(img_copy, joint_text, (10, status_y + 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw brick detection
        if brick_info:
            u, v = brick_info['u'], brick_info['v']
            cv2.circle(img_copy, (u, v), 8, (0, 255, 0), -1)
            cv2.line(img_copy, (u - 15, v), (u + 15, v), (0, 255, 0), 3)
            cv2.line(img_copy, (u, v - 15), (u, v + 15), (0, 255, 0), 3)
            
            cv2.putText(img_copy, f"BRICK ({u}, {v})", (u + 20, v - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_copy, f"Depth: {brick_info['depth']}mm", (u + 20, v + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img_copy, "NO BRICK DETECTED", (10, status_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return img_copy
    
    def run(self):
        """Main data collection loop"""
        print("🎯 TRAJECTORY DATA COLLECTION SYSTEM - CSV OUTPUT")
        print("="*70)
        print("📋 SIMPLIFIED DATA COLLECTION - JOINTS ONLY")
        print("="*70)
        
        # Start trajectory recording thread
        recording_thread = threading.Thread(target=self.trajectory_recording_loop)
        recording_thread.daemon = True
        recording_thread.start()
        
        try:
            while True:
                ret, rgb_frame = self.rgb_cap.read()
                
                if ret:
                    # Detect brick
                    brick_info = self.detect_brick(rgb_frame)
                    
                    # Create display frame
                    display_frame = self.create_display_frame(rgb_frame, brick_info)
                    
                    # Show frame
                    cv2.imshow("Trajectory Data Collection - CSV Output", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r') or key == ord('R'):
                    self.set_rest_position()
                
                elif key == ord(' '):
                    if not self.recording:
                        if brick_info:
                            self.start_recording(brick_info)
                        else:
                            print("❌ No brick detected!")
                    else:
                        self.stop_recording()
                
                elif key == ord('n') or key == ord('N'):
                    self.finish_episode()
                
                elif key == ord('c') or key == ord('C'):
                    self.reset_system()
                
                elif key == ord('s') or key == ord('S'):
                    self.save_csv_data()
                
                elif key == 27:  # ESC
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            self.cleanup()
    
    def trajectory_recording_loop(self):
        """Background thread for recording trajectory steps"""
        while self.running:
            if self.recording:
                self.record_trajectory_step()
                time.sleep(1.0 / self.recording_frequency)
            else:
                time.sleep(0.1)
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🛑 Cleaning up...")
        self.running = False
        self.save_csv_data()
        
        try:
            if hasattr(self, 'depth_stream'):
                self.depth_stream.stop()
            if hasattr(self, 'rgb_cap'):
                self.rgb_cap.release()
            if hasattr(self, 'dev'):
                self.dev.close()
            if hasattr(self, 'ser') and self.ser:
                self.ser.close()
            openni2.unload()
            cv2.destroyAllWindows()
        except:
            pass
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🎯 TRAJECTORY DATA COLLECTION - CSV OUTPUT ONLY")
    print("="*80)
    
    # Set your model paths
    yolo_model_path = r"C:\Users\Administrator\Documents\Simple_CNN_MLP\brick_bowl_yolo\models\brick_bowl_early_stop_20250713_211748\weights\best.pt"
    serial_port = 'COM5'
    
    # Verify models exist
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model not found: {yolo_model_path}")
        return
    
    print(f"🤖 YOLO Model: {yolo_model_path}")
    print(f"🔌 Serial Port: {serial_port}")
    print("="*80)
    
    # Initialize and run collector
    collector = SimplifiedTrajectoryDataCollector(yolo_model_path, serial_port)
    collector.run()

if __name__ == "__main__":
    main()