# Eye-tracking-mouse-control-system
Integration of Eye tracking in assistive technologies: A new era for physical disability support
Modules and Libraries
cv2: For video capture and image processing.
mediapipe: To detect and track facial landmarks.
pyautogui: To control the mouse pointer programmatically.
numpy: For numerical computations like mean and matrix operations.
deque: A double-ended queue for storing and smoothing gaze data.
time: To handle timestamps for actions like blinks.
pyttsx3: For text-to-speech feedback to the user.
Class: GazeMouseController
This class encapsulates the entire logic for gaze-based mouse control. Here's a breakdown of its components:

1. Initialization (__init__)
The constructor initializes:

Mediapipe Face Mesh: For facial landmark detection.
Text-to-Speech Engine: Provides feedback, e.g., "System paused".
Screen Dimensions: Using pyautogui.size() for mouse movement calculations.
Eye and Iris Landmarks: Predefined indices for left and right eye contours and irises.
Smoothing Buffers: deque objects to store recent gaze positions for smoothing.
Kalman Filters: Used for noise reduction in gaze data.
Blink Detection: Tracks blinks, including thresholds and timestamps.
Visualization Parameters: Colors, gaze direction indicators, and heatmap.
Adaptive Speed: Allows for dynamic mouse speed based on gaze distance.

2. Kalman Filter Setup (setup_kalman_filters)
This method configures the Kalman filters for smoothing the gaze coordinates.

Kalman filters are mathematical models that predict the next position based on current and past data, reducing noise and fluctuations in input signals.

3. Gaze and Iris Processing
apply_kalman_filter: Applies the Kalman filter to smooth coordinates.
get_iris_position: Determines the iris's position in normalized coordinates (values between 0 and 1) and calculates its center point for visualization.

4. Smoothing and Gaze Processing
smooth_coordinates: Combines Kalman filtering with weighted moving averages to further smooth the gaze position. A queue stores recent coordinates, and weighted averages prioritize newer values.

5. Blink Detection
calculate_eye_aspect_ratio: Calculates the ratio of eye height to width to detect blinks. A closed eye has a low aspect ratio.
detect_blinks: Uses the aspect ratio to:
Detect normal blinks.
Perform actions based on blink duration (e.g., single blink for left-click, long blink to pause/resume).
Track triple blinks for special actions like double-clicking.

6. Mouse Movement and Actions
move_mouse:
Calculates the direction and speed for mouse movement based on gaze.
Uses the history of gaze positions to smooth motion.
Applies momentum and friction to create natural movement.
Ensures mouse stays within screen bounds.
calculate_adaptive_speed: Dynamically adjusts mouse speed based on how far the gaze is from the screen's center.

7. Feedback and Visualization
speak_feedback: Provides auditory feedback with a cooldown to avoid excessive speech.
Gaze Visualizations:
Draws eye contours and iris centers.
Shows gaze direction using arrows.
Displays heatmaps for areas frequently gazed at.

8. Running the System (run)
The main loop:

Captures a frame from the webcam.
Processes the frame to detect facial landmarks.
Tracks the irises and smooths their positions.
Moves the mouse based on gaze direction.
Detects and handles blinks for clicks or system actions.
Displays visualizations, including gaze directions and blink indicators.

Key Features
Gaze-based Mouse Movement: Uses eye tracking to control the pointer.
Blink-based Clicking:
Short blink: Left click.
Long blink: Pause/resume.
Triple blink: Double-click.
Kalman Filtering: Smoothens noisy gaze data for more precise control.
Adaptive Mouse Speed: Dynamically adjusts speed based on gaze movement.
Heatmap: Tracks frequently gazed areas on the screen.
Visualization: Provides real-time feedback for gaze and blinks.

Enhancements
Momentum and Friction: Makes the mouse movement smoother and more natural.
Text-to-Speech Feedback: Provides user-friendly interaction feedback.
Pause/Resume Feature: Toggle control system with long blinks.

Potential Use Cases
Accessibility for users with physical disabilities.
Hands-free computing for multitasking environments.
Gaming or virtual reality applications.

Limitations
Lighting Sensitivity: Performance may degrade in poor lighting.
Accuracy: Limited by webcam quality and gaze tracking precision.
Fatigue: Extended use might cause eye strain.
Environment Sensitivity: Background distractions can affect performance.
