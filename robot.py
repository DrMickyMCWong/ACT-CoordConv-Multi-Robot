import json
import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
from enum import Enum, auto

class MotorControlType(Enum):
    POSITION_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()

class Robot:
    def __init__(self, device_name: str, baudrate=115200) -> None:
        """Initialize robot with serial communication"""
        self.serial_port = device_name  # Use device_name as serial port
        self.baudrate = baudrate
        self.follower_mac = "FC:E8:C0:F8:D2:38"  # Store follower MAC address
        
        # Initialize serial connection
        self.ser = None
        self.setup_serial()
        
        # Data management
        self.current_joints = np.zeros(6)  # Default joint values
        self.motor_control_state = MotorControlType.UNKNOWN
        self.data_lock = threading.Lock()
        
        # Background reading thread
        self.running = False
        self.reader_thread = None
        self.start_background_thread()

    def setup_serial(self):
        """Initialize serial communication with arm"""
        print(f"🔌 Setting up serial communication on {self.serial_port}...")
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=0.1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)
            print("✅ Serial communication initialized")
            self.motor_control_state = MotorControlType.POSITION_CONTROL
            
        except Exception as e:
            print(f"❌ Serial setup failed: {e}")
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            print(f"🔍 Available ports: {available_ports}")
            self.ser = None
            
    def start_background_thread(self):
        """Start background thread for continuous reading"""
        if self.reader_thread is not None and self.reader_thread.is_alive():
            return  # Thread already running
            
        self.running = True
        self.reader_thread = threading.Thread(target=self.continuous_arm_reading, daemon=True)
        self.reader_thread.start()
        print(f"📡 Started background reading thread for {self.serial_port}")
    
    def stop_background_thread(self):
        """Stop the background reading thread"""
        self.running = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
            print(f"📡 Stopped background reading thread for {self.serial_port}")

    def continuous_arm_reading(self):
        """Continuously read arm position in background thread"""
        while self.running:
            arm_data = self.get_arm_data_immediate()
            if arm_data:
                with self.data_lock:
                    self.current_joints = np.array([
                        arm_data['joints']['base'],
                        arm_data['joints']['shoulder'],
                        arm_data['joints']['elbow'],
                        arm_data['joints']['turn'],
                        arm_data['joints']['roll'],
                        arm_data['joints']['gripper']
                    ])
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
                    
                    # Check if arm_data is a dictionary (not int, string, etc.)
                    if not isinstance(arm_data, dict):
                        return None
                    
                    # Check for message type T:1051 if needed
                    # if arm_data.get('T') != 1051:
                    #     return None
                    
                    # Extract joint angles (radians)
                    base = arm_data.get('b', None)
                    shoulder = arm_data.get('s', None)
                    elbow = arm_data.get('e', None)
                    turn = arm_data.get('t', None)
                    roll = arm_data.get('r', None)
                    gripper = arm_data.get('g', None)  # Added gripper joint
                    
                    # Extract end-effector position (mm)
                    x = arm_data.get('x', None)
                    y = arm_data.get('y', None)
                    z = arm_data.get('z', None)
                
                    # Validate data
                    joints_valid = all(v is not None for v in [base, shoulder, elbow, turn, roll, gripper])
                    position_valid = all(v is not None for v in [x, y, z])
                    
                    if joints_valid:
                        result = {
                            'joints': {
                                'base': base,
                                'shoulder': shoulder,
                                'elbow': elbow,
                                'turn': turn,
                                'roll': roll,
                                'gripper': gripper
                            }
                        }
                        
                        # Add position if available
                        if position_valid:
                            result['position'] = {
                                'x': x,  # mm
                                'y': y,  # mm
                                'z': z   # mm
                            }
                        
                        return result
                        
        except (json.JSONDecodeError, UnicodeDecodeError, serial.SerialException) as e:
            pass
        
        return None

    def read_position(self, tries=2, timeout=0.5):
        """
        Reads the joint positions of the robot.
        Now uses the continuously updated current_joints
        :return: array of joint positions in radians
        """
        # Simply return the latest joint values from the background thread
        with self.data_lock:
            positions = np.copy(self.current_joints)

        # Check if we have valid non-zero data
        if not np.any(positions):
            print(f"⚠️ Warning: Returning zero positions from {self.serial_port}")

        return positions

    def set_initial_position(self):
        """
        Move the robot to a specific starting position for evaluation
        """
        if not self.ser:
            print("No serial connection available")
            return False
            
        # Define the initial joint angles (in radians)
        initial_position = {
            'base': 0.004601942,
            'shoulder': -0.394233062,
            'elbow': 2.604699378,
            'wrist': 0.457126275,
            'roll': 0.001533981,
            'hand': 3.087903326
        }
        
        try:
            # Create and send the move command
            move_command = {
                'T': 102,
                'base': initial_position['base'],
                'shoulder': initial_position['shoulder'],
                'elbow': initial_position['elbow'],
                'wrist': initial_position['wrist'],
                'roll': initial_position['roll'],
                'hand': initial_position['hand'],
                'spd': 0,  # Use a moderate speed for startup movement
                'acc': 10
            }
            
            print("Moving arm to initial position...")
            cmd_str = json.dumps(move_command) + '\n'
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
            
            # Wait for the movement to complete
            # Let's use a longer timeout for the initial movement
            time.sleep(3.0)
            
            # Verify position
            current_pos = self.read_position()
            print(f"Initial position set. Current position: {current_pos}")
            return True
                
        except (serial.SerialException, ValueError) as e:
            print(f"❌ Error setting initial position: {e}")
            return False
    
    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        Since direct velocity reading is not available, returns zeros.
        :return: array of joint velocities in rad/s
        """
        # For now, just return zeros as we don't have velocity data
        return np.zeros(6)
    
    def read_ee_position(self, timeout=0.5):
        """
        Read end-effector position in robot frame (x, y, z in mm).
        For R2RealGen format.
        
        :return: np.array([x, y, z]) in millimeters, or None if not available
        """
        if not self.ser:
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            arm_data = self.get_arm_data_immediate()
            if arm_data and 'position' in arm_data:
                pos = arm_data['position']
                return np.array([pos['x'], pos['y'], pos['z']])
            time.sleep(0.01)
        
        # If no position data available, return None
        return None

    def set_goal_pos(self, action):
        """
        Not used for your teleoperation setup - just a placeholder to maintain compatibility
        :param action: list or numpy array of target joint positions
        """
        # {"T":102,"base":0,"shoulder":0,"elbow":1.57,"wrist":0,"roll":0,"hand":1.57,"spd":0,"acc":10}
        try:
            move_command = {
                'T': 102,
                'base': action[0],
                'shoulder': action[1],
                'elbow': action[2],
                'wrist': action[3],
                'roll': action[4],
                'hand': action[5],
                'spd': 0,
                'acc': 10
            }
        
            cmd_str = json.dumps(move_command) + '\n'
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
            time.sleep(0.01)
            
        except (serial.SerialException, ValueError) as e:
            print(f"❌ Error enabling torque: {e}")
            

    def set_trigger_torque(self):
        """
        Enable torque on the leader arm to allow control
        Implementation based on CMD_TORQUE_CTRL (210)
        """
        if not self.ser:
            print("No serial connection available")
            return
        
        # Check if this is the leader arm (typically /dev/ttyUSB1)
        if '/dev/ttyUSB1' not in self.serial_port:
            print(f"Not a leader arm (port: {self.serial_port})")
            return
        
        try:
            # Step 1: Enable torque using factory JSON command format
            # {"T":210,"cmd":1} - Torque ON
            torque_command = {
                'T': 210,
                'cmd': 1
            }

            cmd_str = json.dumps(torque_command) + '\n'
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
            print("✅ Torque enabled on leader arm")
            self.motor_control_state = MotorControlType.POSITION_CONTROL
            
            # Step 2: Remove follower from ESP-NOW peer list
            # {"T":304,"mac":"FC:E8:C0:F8:D2:38"}
            remove_follower_command = {
                    'T': 304,
                    'mac': self.follower_mac
            }
                
            cmd_str = json.dumps(remove_follower_command) + '\n'
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
            print(f"✅ Removed follower {self.follower_mac} from ESP-NOW peer list")
            time.sleep(0.5)
            
        except (serial.SerialException, ValueError) as e:
            print(f"❌ Error enabling torque: {e}")

    def _disable_torque(self):
        """
        Disable torque on the arm using factory JSON command format
        Only implemented for the leader arm
        Implementation based on CMD_TORQUE_CTRL (210)
        """
        if not self.ser:
            print("No serial connection available")
            return
            
        # Check if this is the leader arm (typically /dev/ttyUSB1)
        if '/dev/ttyUSB1' in self.serial_port:
            try:
                # {"T":210,"cmd":0} - Torque OFF
                torque_command = {
                    'T': 210,
                    'cmd': 0
                }

                cmd_str = json.dumps(torque_command) + '\n'
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
                print("✅ Torque disabled on leader arm")
                self.motor_control_state = MotorControlType.DISABLED
                time.sleep(0.5)
                
                # Step 2: Add follower to ESP-NOW peer list
                # {"T":303,"mac":"FC:E8:C0:F8:D2:38"}
                add_follower_command = {
                    'T': 303,
                    'mac': self.follower_mac
                }

                cmd_str = json.dumps(add_follower_command) + '\n'
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
                print(f"✅ Added follower {self.follower_mac} to ESP-NOW peer list")
                time.sleep(0.5)
                
                # Step 3: Set ESP-NOW to flow-leader (single) mode
                # {"T":301,"mode":2}
                esp_now_mode_command = {
                    'T': 301,
                    'mode': 2  # flow-leader (single)
                }
                
                cmd_str = json.dumps(esp_now_mode_command) + '\n'
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
                print("✅ Set ESP-NOW to flow-leader (single) mode")
                time.sleep(0.5)
                
                print("👥 Leader-follower pairing complete")

            except (serial.SerialException, ValueError) as e:
                print(f"❌ Error disabling torque: {e}")
        else:
            print(f"📝 Torque control only for leader arm (current: {self.serial_port})")
            
    def __del__(self):
        """Clean up when the object is destroyed"""
        self.stop_background_thread()
        if self.ser:
            self.ser.close()