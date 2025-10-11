import cv2
import numpy as np
import os
import glob
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe-silicon (for M1/M2 Mac)")

if not MEDIAPIPE_AVAILABLE:
    print("Using fallback OpenCV face tracker")
    exit()

# Load expression images
EXPRESSION_IMAGES = {}
EXPRESSION_SIZE = (400, 400)  # Standard size for all expression images
nailong_dir = "nailong"

# Create black square for neutral expression
BLACK_SQUARE = np.zeros((EXPRESSION_SIZE[1], EXPRESSION_SIZE[0], 3), dtype=np.uint8)

# Expression name mapping
EXPRESSION_NAMES = {
    "1.jpeg": "Angry",
    "2.jpg": "Tongue Out",
    "3.jpg": "Shocked",
    "4.png": "Smiling"
}

if os.path.exists(nailong_dir):
    for img_file in ["1.jpeg", "2.jpg", "3.jpg", "4.png"]:
        img_path = os.path.join(nailong_dir, img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # For 1.jpeg, fill the entire window maintaining aspect ratio (crop to fill)
                if img_file == "1.jpeg":
                    h, w = img.shape[:2]
                    aspect = w / h
                    target_aspect = EXPRESSION_SIZE[0] / EXPRESSION_SIZE[1]
                    
                    # Scale to fill (larger dimension)
                    if aspect > target_aspect:
                        # Image is wider, scale by height
                        new_h = EXPRESSION_SIZE[1]
                        new_w = int(new_h * aspect)
                    else:
                        # Image is taller, scale by width
                        new_w = EXPRESSION_SIZE[0]
                        new_h = int(new_w / aspect)
                    
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    # Crop center to fit square
                    y_offset = (new_h - EXPRESSION_SIZE[1]) // 2
                    x_offset = (new_w - EXPRESSION_SIZE[0]) // 2
                    img_cropped = img_resized[y_offset:y_offset+EXPRESSION_SIZE[1], 
                                              x_offset:x_offset+EXPRESSION_SIZE[0]]
                    
                    EXPRESSION_IMAGES[img_file] = img_cropped
                else:
                    # Resize to standard size while maintaining aspect ratio
                    h, w = img.shape[:2]
                    aspect = w / h
                    if aspect > 1:
                        new_w = EXPRESSION_SIZE[0]
                        new_h = int(new_w / aspect)
                    else:
                        new_h = EXPRESSION_SIZE[1]
                        new_w = int(new_h * aspect)
                    
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    # Create a canvas and center the image
                    canvas = np.zeros((EXPRESSION_SIZE[1], EXPRESSION_SIZE[0], 3), dtype=np.uint8)
                    y_offset = (EXPRESSION_SIZE[1] - new_h) // 2
                    x_offset = (EXPRESSION_SIZE[0] - new_w) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
                    
                    EXPRESSION_IMAGES[img_file] = canvas
                print(f"Loaded {img_file}")

if not EXPRESSION_IMAGES:
    print("Warning: No expression images found in nailong directory")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs for a cleaner look
FACE_CONNECTIONS_STYLE = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White color
    thickness=1,
    circle_radius=1
)

CONTOUR_STYLE = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White
    thickness=2,
    circle_radius=1
)

IRIS_STYLE = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White
    thickness=2,
    circle_radius=1
)

LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White
    thickness=1,
    circle_radius=1
)

# Transparency for overlays
MESH_ALPHA = 0.25  # Adjust this value (0.0 to 1.0) for more/less transparency
CONTOUR_ALPHA = 0.25  # Transparency for contours and features

# Configure Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("✨ Face Tracker Started!")
print("Press 'q' to quit, 's' to toggle styles")

# Visualization mode
viz_mode = 0  # 0: Full mesh, 1: Contours only, 2: Minimal

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_expression(face_landmarks):
    """
    Detect facial expression based on landmark positions
    Returns: image filename or None
    """
    # Key landmark indices
    # Mouth
    UPPER_LIP_TOP = 13
    LOWER_LIP_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_CENTER = 0
    LOWER_LIP_CENTER = 17
    
    # Eyes
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    # Eyebrows
    LEFT_EYEBROW_INNER = 336
    LEFT_EYEBROW_OUTER = 285
    RIGHT_EYEBROW_INNER = 107
    RIGHT_EYEBROW_OUTER = 55
    LEFT_EYEBROW_CENTER = 300
    RIGHT_EYEBROW_CENTER = 70
    
    # Smile/mouth corners
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    
    landmarks = face_landmarks.landmark
    
    # Calculate mouth aspect ratio (height / width)
    mouth_height = calculate_distance(landmarks[13], landmarks[14])
    mouth_width = calculate_distance(landmarks[61], landmarks[291])
    mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
    
    # Calculate eye openness
    left_eye_height = calculate_distance(landmarks[159], landmarks[145])
    right_eye_height = calculate_distance(landmarks[386], landmarks[374])
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    
    # Calculate mouth openness (vertical distance)
    mouth_open = calculate_distance(landmarks[13], landmarks[14])
    
    # Detect tongue out (lower lip pushed down significantly)
    # Use multiple indicators for better tongue detection
    tongue_indicator = calculate_distance(landmarks[14], landmarks[17])
    lower_lip_bottom = landmarks[14].y
    upper_lip_top = landmarks[13].y
    
    # Calculate eyebrow position (for angry detection)
    # Check multiple eyebrow points for better frown detection
    left_eyebrow_inner_y = landmarks[336].y
    right_eyebrow_inner_y = landmarks[107].y
    left_eyebrow_center_y = landmarks[300].y
    right_eyebrow_center_y = landmarks[70].y
    
    left_eye_top_y = landmarks[159].y
    right_eye_top_y = landmarks[386].y
    
    # Distance between inner eyebrows and eyes (smaller = furrowed)
    eyebrow_to_eye_left = left_eye_top_y - left_eyebrow_inner_y
    eyebrow_to_eye_right = right_eye_top_y - right_eyebrow_inner_y
    avg_eyebrow_distance = (eyebrow_to_eye_left + eyebrow_to_eye_right) / 2
    
    # Eyebrow angle/slant for angry (inner brows lower than outer)
    left_eyebrow_slant = left_eyebrow_inner_y - landmarks[285].y  # inner vs outer
    right_eyebrow_slant = right_eyebrow_inner_y - landmarks[55].y
    eyebrow_frown_angle = (left_eyebrow_slant + right_eyebrow_slant) / 2
    
    # Mouth corners position for smile detection
    left_corner_y = landmarks[61].y
    right_corner_y = landmarks[291].y
    mouth_corners_avg_y = (left_corner_y + right_corner_y) / 2
    upper_lip_y = landmarks[13].y
    lower_lip_y = landmarks[14].y
    mouth_center_y = (upper_lip_y + lower_lip_y) / 2
    
    # Smile: corners should be ABOVE the mouth center (corners pulled up)
    smile_indicator = mouth_center_y - mouth_corners_avg_y
    
    # Frown: corners below upper lip
    frown_indicator = mouth_corners_avg_y - upper_lip_y
    
    # Expression detection with refined thresholds
    # 3.jpg: Mouth wide open with eyes wide open
    if mouth_aspect_ratio > 0.35 and avg_eye_height > 0.025:
        return "3.jpg"
    
    # 2.jpg: Tongue out (EASIER DETECTION - relaxed thresholds)
    # Detect mouth opening + lower lip extension
    if mouth_aspect_ratio > 0.15 and tongue_indicator > 0.015:
        return "2.jpg"
    
    # 1.jpeg: Angry face - VERY SENSITIVE detection for furrowed eyebrows
    # Just need ANY eyebrow lowering or furrowing motion
    if eyebrow_frown_angle > 0.003 or avg_eyebrow_distance < 0.045:
        return "1.jpeg"
    
    # 4.png: Smiling - mouth corners pulled UP (not open mouth)
    # Corners are higher than mouth center, minimal mouth opening
    if smile_indicator > 0.008 and mouth_aspect_ratio < 0.2:
        return "4.png"
    
    return None

def add_glow_effect(frame, points, color, radius=5):
    """Add a subtle glow effect to points"""
    overlay = frame.copy()
    for point in points:
        cv2.circle(overlay, point, radius, color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def draw_minimal_landmarks(frame, face_landmarks, h, w):
    """Draw minimal, aesthetic landmarks"""
    # Key facial feature indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_EYEBROW = [276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [46, 53, 52, 65, 55]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
    NOSE_TIP = [1, 2]
    
    # Draw eye contours with glow
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
        add_glow_effect(frame, points, (0, 200, 200), 3)
    
    # Draw eyebrows
    for brow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
        points = []
        for idx in brow_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], False, (255, 200, 100), 2, cv2.LINE_AA)
    
    # Draw lips
    lip_points = []
    for idx in LIPS_OUTER:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        lip_points.append((x, y))
    
    lip_points = np.array(lip_points, dtype=np.int32)
    cv2.polylines(frame, [lip_points], True, (180, 120, 255), 2, cv2.LINE_AA)
    add_glow_effect(frame, lip_points, (150, 100, 200), 3)
    
    # Draw nose tip
    for idx in NOSE_TIP:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, (100, 255, 100), -1)

current_expression_img = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Add subtle gradient overlay for better aesthetics
    overlay = frame.copy()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect face landmarks
    results = face_mesh.process(rgb_frame)
    
    # Detect expression and show corresponding image
    detected_expression = None
    
    # Draw face mesh annotations on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect expression
            detected_expression = detect_expression(face_landmarks)
            
            if viz_mode == 0:  # Full mesh
                # Create a transparent overlay for the mesh
                mesh_overlay = frame.copy()
                
                # Draw tesselation with custom style on overlay
                mp_drawing.draw_landmarks(
                    image=mesh_overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=FACE_CONNECTIONS_STYLE
                )
                
                # Blend the mesh overlay with the original frame
                cv2.addWeighted(mesh_overlay, MESH_ALPHA, frame, 1 - MESH_ALPHA, 0, frame)
                
                # Create overlay for contours
                feature_overlay = frame.copy()
                
                # Draw contours on overlay
                mp_drawing.draw_landmarks(
                    image=feature_overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=CONTOUR_STYLE
                )
                
                # Blend the feature overlay
                cv2.addWeighted(feature_overlay, CONTOUR_ALPHA, frame, 1 - CONTOUR_ALPHA, 0, frame)
                
            elif viz_mode == 1:  # Contours only
                # Create overlay for contours
                contour_overlay = frame.copy()
                
                mp_drawing.draw_landmarks(
                    image=contour_overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=CONTOUR_STYLE
                )
                
                # Blend with transparency
                cv2.addWeighted(contour_overlay, CONTOUR_ALPHA, frame, 1 - CONTOUR_ALPHA, 0, frame)
            
            else:  # Minimal aesthetic mode
                draw_minimal_landmarks(frame, face_landmarks, h, w)
    
    # Show expression image window (always visible)
    if detected_expression and detected_expression in EXPRESSION_IMAGES:
        if current_expression_img != detected_expression:
            current_expression_img = detected_expression
        cv2.imshow('Expression', EXPRESSION_IMAGES[detected_expression])
    else:
        # Show black square when no expression detected
        current_expression_img = None
        cv2.imshow('Expression', BLACK_SQUARE)
    
    # Add overlay text (no background)
    # Status text with style
    mode_names = ["FULL MESH", "CONTOURS", "MINIMAL"]
    status_text = f"Mode: {mode_names[viz_mode]}"
    
    # Add black outline for better readability
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (100, 255, 100), 2, cv2.LINE_AA)
    
    # Expression name display
    if detected_expression and detected_expression in EXPRESSION_NAMES:
        expression_text = f"Expression: {EXPRESSION_NAMES[detected_expression]}"
        cv2.putText(frame, expression_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, expression_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Instructions
    cv2.putText(frame, "Press 'S' to switch | 'Q' to quit", (w - 450, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "Press 'S' to switch | 'Q' to quit", (w - 450, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Face Tracker', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        viz_mode = (viz_mode + 1) % 3
        print(f"Switched to mode: {mode_names[viz_mode]}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Face tracker closed.")

