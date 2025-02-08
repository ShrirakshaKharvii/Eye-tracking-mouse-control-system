import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time
import pyttsx3

class GazeMouseController:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  
        self.last_speech_time = 0
        self.speech_cooldown = 2.0  
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Landmarks for iris, eyes, and eye contours
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Upper and lower eye landmarks 
        self.LEFT_EYE_UPPER = [386, 374, 373, 390, 388, 387]
        self.LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380]
        self.RIGHT_EYE_UPPER = [159, 145, 144, 163, 161, 160]
        self.RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153]
        
        # Movement settings
        self.movement_threshold = 0.015
        self.mouse_speed = 20
        self.click_cooldown = 1.0
        
        # Smoothing buffers
        self.smooth_points_left = deque(maxlen=7)
        self.smooth_points_right = deque(maxlen=7)
        
        # Blink detection parameters
        self.blink_threshold = 0.2
        self.last_blink_time = 0
        self.blink_duration_threshold = 0.3
        self.long_blink_threshold = 1.0  
        self.triple_blink_interval = 1.0 
        self.blink_counter = 0
        self.last_click_time = 0
        self.is_paused = False
        self.blink_start_time = 0
        self.left_eye_closed = False
        self.right_eye_closed = False
        
        # Visualization settings
        self.gaze_visualization_length = 60
        self.colors = {
            'left_iris': (0, 255, 0),
            'right_iris': (0, 255, 255),
            'eye_contour': (255, 255, 255),
            'gaze_direction': (0, 0, 255),
            'center_point': (255, 0, 0),
            'text': (255, 255, 255),
            'blink_indicator': (255, 165, 0),
            'grid': (128, 128, 128),
            'heatmap': (0, 140, 255)
        }
        
        # Gaze heatmap
        self.heatmap = np.zeros((10, 10), dtype=np.float32)
        self.heatmap_decay = 0.95

        # Add adaptive movement speed
        self.base_mouse_speed = 15
        self.max_mouse_speed = 45
        self.speed_multiplier = 1.0
        self.distance_threshold = 0.1  
        
        # Add momentum tracking
        self.velocity_x = 0
        self.velocity_y = 0
        self.momentum = 0.8  
        self.friction = 0.2 
        
        # Add noise reduction
        self.min_movement = 0.02 
        self.gaze_history = deque(maxlen=5) 
        
        # Add precision mode
        self.precision_mode = False
        self.precision_speed_multiplier = 0.3
        
        # Add focus zones
        self.focus_zones = {
            'center': {'x': (0.4, 0.6), 'y': (0.4, 0.6)},
            'corners': [
                {'x': (0.0, 0.2), 'y': (0.0, 0.2)},  
                {'x': (0.8, 1.0), 'y': (0.0, 0.2)}, 
                {'x': (0.0, 0.2), 'y': (0.8, 1.0)},  
                {'x': (0.8, 1.0), 'y': (0.8, 1.0)}  
            ]
        }

    def calculate_adaptive_speed(self, distance_from_center):
        """Calculate mouse speed based on gaze distance from center"""
        if distance_from_center < self.distance_threshold:
            return self.base_mouse_speed
        
        speed_factor = min(distance_from_center / self.distance_threshold, 3.0)
        return min(self.base_mouse_speed * speed_factor, self.max_mouse_speed)

    def apply_momentum(self, target_x, target_y, current_x, current_y):
        """Apply momentum and friction to mouse movement"""
        # Update velocity with new target position
        self.velocity_x = (self.velocity_x * self.momentum + 
                         (target_x - current_x) * (1 - self.momentum))
        self.velocity_y = (self.velocity_y * self.momentum + 
                         (target_y - current_y) * (1 - self.momentum))
        
        # Apply friction
        self.velocity_x *= (1 - self.friction)
        self.velocity_y *= (1 - self.friction)
        
        return current_x + self.velocity_x, current_y + self.velocity_y

    def check_focus_zone(self, gaze_x, gaze_y):
        """Check if gaze is in a focus zone and adjust speed accordingly"""
        # Check center zone
        center = self.focus_zones['center']
        if (center['x'][0] <= gaze_x <= center['x'][1] and 
            center['y'][0] <= gaze_y <= center['y'][1]):
            self.speed_multiplier = 0.5
            return True
            
        # Check corner zones
        for corner in self.focus_zones['corners']:
            if (corner['x'][0] <= gaze_x <= corner['x'][1] and 
                corner['y'][0] <= gaze_y <= corner['y'][1]):
                self.speed_multiplier = 1.5
                return True
                
        self.speed_multiplier = 1.0
        return False

    def speak_feedback(self, text):
        current_time = time.time()
        if current_time - self.last_speech_time >= self.speech_cooldown:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.last_speech_time = current_time
        
    def calculate_eye_aspect_ratio(self, eye_upper, eye_lower, landmarks, image):
        """Calculate the eye aspect ratio for blink detection"""
        upper_points = np.array([
            np.multiply([landmarks.landmark[idx].x, landmarks.landmark[idx].y],
                       [image.shape[1], image.shape[0]]).astype(int)
            for idx in eye_upper
        ])
        
        lower_points = np.array([
            np.multiply([landmarks.landmark[idx].x, landmarks.landmark[idx].y],
                       [image.shape[1], image.shape[0]]).astype(int)
            for idx in eye_lower
        ])
        
        upper_mean = np.mean(upper_points[:, 1])
        lower_mean = np.mean(lower_points[:, 1])
        eye_height = abs(upper_mean - lower_mean)
        eye_width = np.linalg.norm(upper_points[0] - upper_points[-1])
        
        ear = eye_height / eye_width
        return ear
        
    def detect_blinks(self, landmarks, frame, current_time):
        """Blink detection with left/right eye distinction and triple blink"""
        left_ear = self.calculate_eye_aspect_ratio(
            self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER, landmarks, frame)
        right_ear = self.calculate_eye_aspect_ratio(
            self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER, landmarks, frame)
        
        action = None
        both_eyes_closed = (left_ear < self.blink_threshold and 
                           right_ear < self.blink_threshold)
        left_eye_only = (left_ear < self.blink_threshold and 
                        right_ear >= self.blink_threshold)
        
        # Check for pause/resume (long blink)
        if both_eyes_closed:
            if self.blink_start_time == 0:
                self.blink_start_time = current_time
            elif current_time - self.blink_start_time >= self.long_blink_threshold:
                self.is_paused = not self.is_paused
                self.speak_feedback("System " + 
                                  ("paused" if self.is_paused else "resumed"))
                self.blink_start_time = 0
                return None, (left_ear, right_ear)
        else:
            if self.blink_start_time != 0:
                blink_duration = current_time - self.blink_start_time
                
                # Handle different blink types
                if blink_duration < self.blink_duration_threshold:
                    if current_time - self.last_blink_time < self.triple_blink_interval:
                        self.blink_counter += 1
                        if self.blink_counter == 3:
                            action = "double_click"
                            self.speak_feedback("Double click")
                            self.blink_counter = 0
                    else:
                        self.blink_counter = 1
                        action = "left_click"
                        self.speak_feedback("Left click")
                    
                    self.last_blink_time = current_time
            
            self.blink_start_time = 0
        
        # Check for right click
        if left_eye_only and current_time - self.last_click_time > self.click_cooldown:
            action = "right_click"
            self.speak_feedback("Right click")
        
        return action, (left_ear, right_ear)
    
    def handle_mouse_actions(self, action, current_time):
        """Execute mouse actions with click types"""
        if current_time - self.last_click_time < self.click_cooldown:
            return
            
        if action == "left_click":
            pyautogui.click(button='left')
            self.last_click_time = current_time
        elif action == "right_click":
            pyautogui.click(button='right')
            self.last_click_time = current_time
        elif action == "double_click":
            pyautogui.doubleClick()
            self.last_click_time = current_time

    def draw_visualization_grid(self, frame):
        """Draw a grid overlay for visualization"""
        h, w = frame.shape[:2]
        cells = 8
        
        # Draw vertical lines
        for i in range(1, cells):
            x = int(w * i / cells)
            cv2.line(frame, (x, 0), (x, h), self.colors['grid'], 1)
            
        # Draw horizontal lines
        for i in range(1, cells):
            y = int(h * i / cells)
            cv2.line(frame, (0, y), (w, y), self.colors['grid'], 1)

    def update_heatmap(self, gaze_x, gaze_y):
        """Update gaze heatmap"""
        x_idx = min(int(gaze_x * 10), 9)
        y_idx = min(int(gaze_y * 10), 9)
        self.heatmap *= self.heatmap_decay
        self.heatmap[y_idx, x_idx] += 1

    def draw_heatmap(self, frame):
        """Draw the gaze heatmap overlay"""
        h, w = frame.shape[:2]
        heatmap_vis = cv2.resize(self.heatmap, (w, h))
        heatmap_vis = cv2.normalize(heatmap_vis, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_vis = heatmap_vis.astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        frame_with_heatmap = cv2.addWeighted(frame, 0.7, heatmap_vis, 0.3, 0)
        return frame_with_heatmap

    def draw_eye_state_indicator(self, frame, left_ear, right_ear):
        """Draw indicators for eye state"""
        h, w = frame.shape[:2]
        padding = 10
        bar_height = 50
        bar_width = 20
        
        # Left eye indicator
        left_height = int(bar_height * min(left_ear * 3, 1))
        cv2.rectangle(frame, 
                     (padding, h - padding - bar_height),
                     (padding + bar_width, h - padding),
                     self.colors['blink_indicator'], 2)
        cv2.rectangle(frame,
                     (padding, h - padding - left_height),
                     (padding + bar_width, h - padding),
                     self.colors['blink_indicator'], -1)
        
        # Right eye indicator
        right_height = int(bar_height * min(right_ear * 3, 1))
        cv2.rectangle(frame, 
                     (w - padding - bar_width, h - padding - bar_height),
                     (w - padding, h - padding),
                     self.colors['blink_indicator'], 2)
        cv2.rectangle(frame,
                     (w - padding - bar_width, h - padding - right_height),
                     (w - padding, h - padding),
                     self.colors['blink_indicator'], -1)

    def get_iris_position(self, iris_landmarks, image):
        """Calculate normalized iris position and center point"""
        mesh_points = np.array([
            np.multiply([p.x, p.y], [image.shape[1], image.shape[0]]).astype(int)
            for p in iris_landmarks
        ])
        
        center = np.mean(mesh_points, axis=0).astype(int)
        normalized_x = center[0] / image.shape[1]
        normalized_y = center[1] / image.shape[0]
        
        return normalized_x, normalized_y, center, mesh_points

    def draw_eye_contour(self, landmarks, eye_indices, frame):
        """Draw eye contour"""
        points = np.array([
            np.multiply([landmarks.landmark[idx].x, landmarks.landmark[idx].y],
                       [frame.shape[1], frame.shape[0]]).astype(int)
            for idx in eye_indices
        ])
        cv2.polylines(frame, [points], True, self.colors['eye_contour'], 1)

    def draw_gaze_direction(self, center, gaze_x, gaze_y, frame):
        """Draw gaze direction indicator"""
        direction_x = int((gaze_x - 0.5) * self.gaze_visualization_length * 2)
        direction_y = int((gaze_y - 0.5) * self.gaze_visualization_length * 2)
        end_point = (
            center[0] + direction_x,
            center[1] + direction_y
        )
        cv2.arrowedLine(frame, tuple(center), end_point, 
                        self.colors['gaze_direction'], 2)

    def smooth_coordinates(self, x, y, smooth_buffer):
        """Apply smoothing to coordinates"""
        smooth_buffer.append((x, y))
        if len(smooth_buffer) < smooth_buffer.maxlen:
            return x, y
        
        smooth_x = sum(p[0] for p in smooth_buffer) / len(smooth_buffer)
        smooth_y = sum(p[1] for p in smooth_buffer) / len(smooth_buffer)
        return smooth_x, smooth_y

    def calculate_combined_gaze(self, left_gaze, right_gaze):
        """Calculate combined gaze direction from both eyes"""
        return (
            (left_gaze[0] + right_gaze[0]) / 2,
            (left_gaze[1] + right_gaze[1]) / 2
        )

    def move_mouse(self, gaze_x, gaze_y):
        if self.is_paused:
            return
            
        current_x, current_y = pyautogui.position()
        
        self.gaze_history.append((gaze_x, gaze_y))
        if len(self.gaze_history) < self.gaze_history.maxlen:
            return
            
        smooth_x = sum(p[0] for p in self.gaze_history) / len(self.gaze_history)
        smooth_y = sum(p[1] for p in self.gaze_history) / len(self.gaze_history)
        
        distance_from_center = np.sqrt((smooth_x - 0.5)**2 + (smooth_y - 0.5)**2)
        
        if distance_from_center < self.min_movement:
            return
            
        self.check_focus_zone(smooth_x, smooth_y)
        
        base_speed = self.calculate_adaptive_speed(distance_from_center)
        actual_speed = base_speed * self.speed_multiplier
        
        if self.precision_mode:
            actual_speed *= self.precision_speed_multiplier
        
        move_x = move_y = 0
        if abs(smooth_x - 0.5) > self.movement_threshold:
            move_x = actual_speed if smooth_x > 0.5 else -actual_speed
        if abs(smooth_y - 0.5) > self.movement_threshold:
            move_y = actual_speed if smooth_y > 0.5 else -actual_speed
        
        new_x, new_y = self.apply_momentum(
            current_x + move_x,
            current_y + move_y,
            current_x,
            current_y
        )
        
        new_x = max(0, min(self.screen_w, new_x))
        new_y = max(0, min(self.screen_h, new_y))
        
        pyautogui.moveTo(new_x, new_y)

    def add_text_overlay(self, frame, left_gaze, right_gaze, combined_gaze, ear_values):
        info_text = [
            f"Left Gaze: ({left_gaze[0]:.2f}, {left_gaze[1]:.2f})",
            f"Right Gaze: ({right_gaze[0]:.2f}, {right_gaze[1]:.2f})",
            f"Combined: ({combined_gaze[0]:.2f}, {combined_gaze[1]:.2f})",
            f"Left EAR: {ear_values[0]:.2f}",
            f"Right EAR: {ear_values[1]:.2f}",
            "Controls:",
            "Single Blink: Left Click",
            "Left Eye Blink: Right Click",
            "Triple Blink: Double Click",
            f"Long Blink: {'Resume' if self.is_paused else 'Pause'}",
            "Press 'q' to quit"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['text'], 2)

    def run(self):
        """Main run loop"""
        cap = cv2.VideoCapture(0)
        self.speak_feedback("Eye control started")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Get current time for blink detection
                current_time = time.time()
                
                # Detect blinks and get eye aspect ratios
                blink_action, ear_values = self.detect_blinks(landmarks, frame, current_time)
                
                # Handle mouse actions based on blinks
                self.handle_mouse_actions(blink_action, current_time)
                
                # Process left eye
                left_iris_landmarks = [landmarks.landmark[i] for i in self.LEFT_IRIS]
                left_gaze_x, left_gaze_y, left_center, left_mesh = self.get_iris_position(left_iris_landmarks, frame)
                
                # Process right eye
                right_iris_landmarks = [landmarks.landmark[i] for i in self.RIGHT_IRIS]
                right_gaze_x, right_gaze_y, right_center, right_mesh = self.get_iris_position(right_iris_landmarks, frame)
                
                # Smooth the coordinates
                smooth_left_x, smooth_left_y = self.smooth_coordinates(
                    left_gaze_x, left_gaze_y, self.smooth_points_left)
                smooth_right_x, smooth_right_y = self.smooth_coordinates(
                    right_gaze_x, right_gaze_y, self.smooth_points_right)
                
                # Calculate combined gaze
                combined_gaze = self.calculate_combined_gaze(
                    (smooth_left_x, smooth_left_y),
                    (smooth_right_x, smooth_right_y)
                )
                
                # Update heatmap
                self.update_heatmap(*combined_gaze)
                
                # Move mouse based on combined gaze
                self.move_mouse(*combined_gaze)
                
                # Draw visualizations
                self.draw_visualization_grid(frame)
                frame = self.draw_heatmap(frame)
                self.draw_eye_contour(landmarks, self.LEFT_EYE, frame)
                self.draw_eye_contour(landmarks, self.RIGHT_EYE, frame)
                
                # Draw iris detection points
                for point in left_mesh:
                    cv2.circle(frame, point, 1, self.colors['left_iris'], -1)
                for point in right_mesh:
                    cv2.circle(frame, point, 1, self.colors['right_iris'], -1)
                
                # Draw center points and gaze directions
                cv2.circle(frame, tuple(left_center), 3, self.colors['center_point'], -1)
                cv2.circle(frame, tuple(right_center), 3, self.colors['center_point'], -1)
                self.draw_gaze_direction(left_center, smooth_left_x, smooth_left_y, frame)
                self.draw_gaze_direction(right_center, smooth_right_x, smooth_right_y, frame)
                
                # Draw eye state indicators
                self.draw_eye_state_indicator(frame, *ear_values)
                
                # Add text overlay
                self.add_text_overlay(frame, 
                                    (smooth_left_x, smooth_left_y),
                                    (smooth_right_x, smooth_right_y),
                                    combined_gaze,
                                    ear_values)
            
            # Display the frame
            cv2.imshow('Eye Controlled Mouse', frame)
            
            # Break loop 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.speak_feedback("Eye control stopped")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.1
    
    try:
        controller = GazeMouseController()
        controller.run()
    except Exception as e:
        print(f"Error occurred: {str(e)}")