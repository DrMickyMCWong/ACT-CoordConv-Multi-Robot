"""
RoArm-M3 Curriculum LSTM Controller - FINAL VERSION
Fixed sequence length issue and added simple GUI
"""
import torch
import torch.nn as nn
import cv2
import serial
import json
import numpy as np
import time
from pathlib import Path
from collections import deque

# Model classes - exactly matching curriculum training script
class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = temperature
        
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height)
        )
        
        self.register_buffer('pos_x', torch.FloatTensor(pos_x))
        self.register_buffer('pos_y', torch.FloatTensor(pos_y))
    
    def forward(self, feature):
        batch_size = feature.size(0)
        feature = feature.view(batch_size, self.channel, -1)
        attention = torch.softmax(feature / self.temperature, dim=-1)
        
        expected_x = torch.sum(self.pos_x.view(-1) * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y.view(-1) * attention, dim=-1, keepdim=True)
        
        expected_xy = torch.stack([expected_x, expected_y], dim=-1)
        return expected_xy.view(batch_size, -1)

class DualCameraCNNSpatialSoftmax(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.spatial_softmax = SpatialSoftmax(128, 128, 16)
        
    def forward(self, x):
        features = self.cnn(x)
        spatial_features = self.spatial_softmax(features)
        return spatial_features

class CurriculumLSTMModel(nn.Module):
    """Curriculum LSTM model - EXACTLY matches training script"""
    def __init__(self, state_dim=6, action_dim=6, lstm_hidden=512, lstm_layers=6, 
                 max_sequence_length=10, curriculum_stage="stage1"):
        super().__init__()
        
        self.curriculum_stage = curriculum_stage
        self.max_sequence_length = max_sequence_length
        
        # Dual CNN + Spatial Softmax
        self.cnn_main = DualCameraCNNSpatialSoftmax()
        self.cnn_secondary = DualCameraCNNSpatialSoftmax()
        
        # Feature dimensions
        spatial_features_per_camera = 32
        total_image_features = spatial_features_per_camera * 2
        self.lstm_input_dim = state_dim + total_image_features
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # Multi-layer LSTM with feature concatenation
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(self.lstm_input_dim if i == 0 else lstm_hidden + total_image_features, 
                       lstm_hidden)
            for i in range(lstm_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(lstm_hidden + total_image_features, action_dim)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def forward(self, seq_img_main, seq_img_secondary, seq_states):
        batch_size, seq_len = seq_img_main.size(0), seq_img_main.size(1)
        
        # Process all images in sequence
        all_img_features = []
        for t in range(seq_len):
            features_main = self.cnn_main(seq_img_main[:, t])
            features_secondary = self.cnn_secondary(seq_img_secondary[:, t])
            combined_features = torch.cat([features_main, features_secondary], dim=1)
            all_img_features.append(combined_features)
        
        # Process through LSTM
        outputs = []
        hidden_states = [None] * self.lstm_layers
        
        for t in range(seq_len):
            current_input = torch.cat([seq_states[:, t], all_img_features[t]], dim=1)
            
            for layer in range(self.lstm_layers):
                if layer == 0:
                    layer_input = current_input
                else:
                    layer_input = torch.cat([hidden_states[layer-1][0], all_img_features[t]], dim=1)
                
                if hidden_states[layer] is None:
                    h = torch.zeros(batch_size, self.lstm_hidden, device=current_input.device)
                    c = torch.zeros(batch_size, self.lstm_hidden, device=current_input.device)
                    hidden_states[layer] = (h, c)
                
                hidden_states[layer] = self.lstm_cells[layer](layer_input, hidden_states[layer])
            
            final_input = torch.cat([hidden_states[-1][0], all_img_features[t]], dim=1)
            output = self.output_layer(final_input)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class RoArmCurriculumController:
    """RoArm-M3 Curriculum LSTM Controller - FINAL VERSION"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model first
        self.load_model(model_path)
        
        # 🔧 FIXED: Force sequence_length to be integer
        self.sequence_length = int(self.sequence_length)
        print(f"🔧 FIXED: sequence_length = {self.sequence_length} (type: {type(self.sequence_length)})")
        
        # Joint ranges
        self.joint_ranges = {
            'b': (-6.28, 6.28), 's': (-3.14, 3.14), 'e': (-3.14, 3.14),
            't': (-3.14, 3.14), 'r': (-6.28, 6.28), 'g': (0.0, 4.0)
        }
        
        # Create buffers AFTER ensuring sequence_length is integer
        self.image_buffer_main = deque(maxlen=self.sequence_length)
        self.image_buffer_secondary = deque(maxlen=self.sequence_length)
        self.state_buffer = deque(maxlen=self.sequence_length)
        
        print(f"🎯 Curriculum Controller Ready!")
        print(f"🎓 Stage: {self.curriculum_stage}")
        print(f"📏 Sequence: {self.sequence_length} frames")
        
        # Setup hardware using SAME METHOD as integration test
        self.setup_cameras_integration_style()
        self.setup_robot_integration_style()
        
        print("✅ Controller ready for Stage 1 testing!")
        
    def load_model(self, model_path):
        """Load curriculum model"""
        print(f"📥 Loading: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 🔧 FIXED: Ensure sequence_length is always integer
        sequence_length = checkpoint.get('sequence_length', 8)
        if 'max_sequence_length' in checkpoint:
            sequence_length = checkpoint['max_sequence_length']
        
        # Force to integer
        sequence_length = int(sequence_length)
            
        state_dim = checkpoint.get('state_dim', 6)
        action_dim = checkpoint.get('action_dim', 6)
        lstm_hidden = checkpoint.get('lstm_hidden', 512)
        lstm_layers = checkpoint.get('lstm_layers', 6)
        curriculum_stage = checkpoint.get('curriculum_stage', 'stage1')
        skill_learned = checkpoint.get('skill_learned', 'reaching_motion')
        
        # Store info
        self.sequence_length = sequence_length  # This is now guaranteed to be int
        self.curriculum_stage = curriculum_stage
        self.skill_learned = skill_learned
        
        print(f"🎓 Detected: {sequence_length} frames, Stage: {curriculum_stage}")
        
        # Create model
        self.model = CurriculumLSTMModel(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            max_sequence_length=sequence_length,
            curriculum_stage=curriculum_stage
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded: ±{checkpoint.get('rmse_deg', 0):.1f}° accuracy")
    
    def find_cameras(self):
        """Find cameras - EXACT COPY from integration test"""
        print("🔍 Detecting cameras...")
        cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(i)
                    print(f"  📷 Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
        
        return cameras
    
    def setup_cameras_integration_style(self):
        """Setup cameras - EXACT COPY from integration test"""
        cameras = self.find_cameras()
        if len(cameras) < 2:
            raise Exception("❌ Need at least 2 cameras!")
        
        cam_indices = cameras[:2]  # Use first two
        print(f"🎥 Setting up cameras {cam_indices[0]} and {cam_indices[1]}...")
        
        self.cap1 = cv2.VideoCapture(cam_indices[0], cv2.CAP_DSHOW)
        self.cap2 = cv2.VideoCapture(cam_indices[1], cv2.CAP_DSHOW)
        
        # Configure cameras - EXACT COPY from integration test
        for cap, name in [(self.cap1, "Camera 1"), (self.cap2, "Camera 2")]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        print("✅ Cameras ready!")
    
    def setup_robot_integration_style(self):
        """Setup robot - EXACT COPY from integration test"""
        LEADER_PORT = "COM6"
        FOLLOWER_PORT = "COM5"
        
        print(f"🤖 Connecting to {LEADER_PORT} and {FOLLOWER_PORT}...")
        
        self.leader = serial.Serial(LEADER_PORT, 115200, timeout=1.0)
        self.follower = serial.Serial(FOLLOWER_PORT, 115200, timeout=1.0)
        
        time.sleep(2.0)  # Same stabilization time as integration test
        
        # Flush buffers - same as integration test
        self.leader.flushInput()
        self.follower.flushInput()
        
        print("✅ Robot connections ready!")
    
    def validate_robot_state(self, state_dict):
        """Validate robot state"""
        if not state_dict or not isinstance(state_dict, dict):
            return False
        
        required_keys = ['b', 's', 'e', 't', 'r', 'g']
        if not all(key in state_dict for key in required_keys):
            return False
        
        try:
            for key in required_keys:
                value = float(state_dict[key])
                min_val, max_val = self.joint_ranges[key]
                if not np.isfinite(value) or value < min_val or value > max_val:
                    return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def extract_6d_state(self, state_dict):
        """Extract 6D state vector"""
        return np.array([
            float(state_dict['b']), float(state_dict['s']), float(state_dict['e']),
            float(state_dict['t']), float(state_dict['r']), float(state_dict['g'])
        ], dtype=np.float32)
    
    def read_robot_state(self):
        """Read robot state from follower"""
        try:
            if self.follower.in_waiting > 0:
                line = self.follower.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    data = json.loads(line)
                    if self.validate_robot_state(data):
                        return self.extract_6d_state(data)
        except:
            pass
        return None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model - CORRECTED TENSOR FORMAT"""
        # Resize to 128x128
        frame = cv2.resize(frame, (128, 128))
        
        # Convert BGR→RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        # 🔧 CRITICAL FIX: Transpose to [C, H, W] format for PyTorch
        # Original: [H, W, C] = [128, 128, 3]
        # Required: [C, H, W] = [3, 128, 128]
        frame = np.transpose(frame, (2, 0, 1))  # [H,W,C] → [C,H,W]
        
        return frame
    
    def add_to_buffers(self, frame1, frame2, state):
        """Add data to buffers"""
        if state is not None:
            self.image_buffer_main.append(self.preprocess_frame(frame1))
            self.image_buffer_secondary.append(self.preprocess_frame(frame2))
            self.state_buffer.append(state)
            return True
        return False
    
    def predict_action(self):
        """Predict action using model"""
        if len(self.state_buffer) < self.sequence_length:
            return None
    
        try:
            with torch.no_grad():
                # Convert to tensors
                seq_img_main = torch.FloatTensor(np.array(list(self.image_buffer_main))).unsqueeze(0).to(self.device)
                seq_img_secondary = torch.FloatTensor(np.array(list(self.image_buffer_secondary))).unsqueeze(0).to(self.device)
                seq_states = torch.FloatTensor(np.array(list(self.state_buffer))).unsqueeze(0).to(self.device)
                
                # Predict
                predicted_actions = self.model(seq_img_main, seq_img_secondary, seq_states)
                action = predicted_actions[0, -1].cpu().numpy()
                
                print(f"🎯 Stage1 → B:{action[0]:.3f} S:{action[1]:.3f} E:{action[2]:.3f} W:{action[3]:.3f} R:{action[4]:.3f} H:{action[5]:.3f}")
                
                return action
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def send_action(self, action):
        """Send T:102 command"""
        try:
            # Clip to safe ranges
            safe_action = [
                float(np.clip(action[0], self.joint_ranges['b'][0], self.joint_ranges['b'][1])),
                float(np.clip(action[1], self.joint_ranges['s'][0], self.joint_ranges['s'][1])),
                float(np.clip(action[2], self.joint_ranges['e'][0], self.joint_ranges['e'][1])),
                float(np.clip(action[3], self.joint_ranges['t'][0], self.joint_ranges['t'][1])),
                float(np.clip(action[4], self.joint_ranges['r'][0], self.joint_ranges['r'][1])),
                float(np.clip(action[5], self.joint_ranges['g'][0], self.joint_ranges['g'][1]))
            ]
            
            # T:102 command
            command = {
                "T": 102,
                "base": safe_action[0], "shoulder": safe_action[1], "elbow": safe_action[2],
                "wrist": safe_action[3], "roll": safe_action[4], "hand": safe_action[5],
                "spd": 0, "acc": 10
            }
            
            # Send
            self.follower.write((json.dumps(command) + '\n').encode())
            return True
            
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False
    
    def create_display(self, frame1, frame2, current_state, predicted_action, status):
        """Create enhanced display with robot state and predictions"""
        # Resize frames for display
        frame1_display = cv2.resize(frame1, (320, 240))
        frame2_display = cv2.resize(frame2, (320, 240))
        
        # Create main display area
        display_width = 680  # Two cameras + margins + info panel
        display_height = 400
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Place camera frames
        display[20:260, 10:330] = frame1_display
        display[20:260, 340:660] = frame2_display
        
        # Camera labels
        cv2.putText(display, "Camera 1", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "Camera 2", (345, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status
        color = (0, 255, 0) if "AUTO" in status else (255, 255, 0) if "BUFFER" in status else (255, 255, 255)
        cv2.putText(display, f"Status: {status}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Stage info
        cv2.putText(display, f"Stage: {self.curriculum_stage}", (10, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display, f"Skill: {self.skill_learned}", (10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Buffer status
        buffer_color = (0, 255, 0) if len(self.state_buffer) >= self.sequence_length else (255, 255, 0)
        cv2.putText(display, f"Buffer: {len(self.state_buffer)}/{self.sequence_length}", (200, 315), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, buffer_color, 1)
        
        # Current robot state
        if current_state is not None:
            cv2.putText(display, "Current State:", (350, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            joint_names = ['B', 'S', 'E', 'W', 'R', 'H']
            for i, (name, value) in enumerate(zip(joint_names, current_state)):
                y_pos = 310 + (i % 3) * 15
                x_pos = 350 + (i // 3) * 120
                cv2.putText(display, f"{name}:{value:.2f}", (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Predicted actions
        if predicted_action is not None:
            cv2.putText(display, "Predicted:", (500, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for i, (name, value) in enumerate(zip(joint_names, predicted_action)):
                y_pos = 310 + (i % 3) * 15
                x_pos = 500 + (i // 3) * 120
                cv2.putText(display, f"{name}:{value:.2f}", (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Controls
        cv2.putText(display, "Controls: 'A'=Auto, 'S'=Stop, 'Q'=Quit, 'R'=Reset", 
                   (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def run(self):
        """Main control loop with enhanced GUI"""
        print("🚀 Starting Stage 1 Curriculum Control")
        print("🎯 Place blue brick in view and press 'A' to start autonomous mode")
        print("Controls:")
        print("  'A' = Start autonomous mode")
        print("  'S' = Stop autonomous mode") 
        print("  'R' = Reset buffers")
        print("  'Q' = Quit")
        
        autonomous = False
        
        try:
            while True:
                # Read frames
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    continue
                
                # Read robot state
                current_state = self.read_robot_state()
                buffer_ready = self.add_to_buffers(frame1, frame2, current_state)
                
                predicted_action = None
                
                # Control logic
                if autonomous and buffer_ready and len(self.state_buffer) >= self.sequence_length:
                    predicted_action = self.predict_action()
                    if predicted_action is not None and self.send_action(predicted_action):
                        status = "🎓 STAGE1 AUTO"
                    else:
                        status = "❌ PREDICTION FAILED"
                        autonomous = False
                elif len(self.state_buffer) < self.sequence_length:
                    status = f"🔄 FILLING BUFFER ({len(self.state_buffer)}/{self.sequence_length})"
                else:
                    status = "⏸️ MANUAL MODE"
                
                # Create and show enhanced display
                display = self.create_display(frame1, frame2, current_state, predicted_action, status)
                cv2.imshow('RoArm Stage 1 Curriculum Controller', display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('a') or key == ord('A'):
                    if len(self.state_buffer) >= self.sequence_length:
                        autonomous = True
                        print("🎓 Stage 1 autonomous mode ACTIVATED!")
                        print("🎯 Robot should start reaching towards blue brick...")
                    else:
                        print(f"⚠️ Need {self.sequence_length} frames in buffer (have {len(self.state_buffer)})")
                        
                elif key == ord('s') or key == ord('S'):
                    autonomous = False
                    print("🛑 Autonomous mode DEACTIVATED")
                    
                elif key == ord('r') or key == ord('R'):
                    # Reset buffers
                    self.image_buffer_main.clear()
                    self.image_buffer_secondary.clear()
                    self.state_buffer.clear()
                    autonomous = False
                    print("🔄 Buffers RESET")
                    
                elif key == ord('q') or key == ord('Q'):
                    print("👋 Quitting...")
                    break
                
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            print("🧹 Cleaning up...")
            self.cap1.release()
            self.cap2.release()
            cv2.destroyAllWindows()
            self.leader.close()
            self.follower.close()
            print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🎓 RoArm Stage 1 Curriculum Controller - FINAL")
    print("🔧 Fixed sequence length integer issue")
    print("🎨 Enhanced GUI with robot state display")
    
    model_path = Path(r"C:\Users\Administrator\Documents\Simple_CNN_MLP\trained_models\best_stage1_curriculum_lstm_model.pth")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Using: {model_path.name}")
    
    try:
        controller = RoArmCurriculumController(model_path)
        controller.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()