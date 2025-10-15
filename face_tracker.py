import cv2
import numpy as np
import os
import sys
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

# ============================================================================
# FOLDER CONFIGURATIONS
# ============================================================================
# Define configurations for each folder with images
# Each configuration includes: directory, expression mappings, and detection settings

FOLDER_CONFIGS = {
    "nailong": {
        "directory": "nailong",
        "description": "Nailong expressions (face detection)",
        "expressions": {
            "1.jpeg": {"name": "Angry", "fill_mode": "crop"},
            "2.jpg": {"name": "Tongue Out", "fill_mode": "fit"},
            "3.jpg": {"name": "Shocked", "fill_mode": "fit"},
            "4.png": {"name": "Smiling", "fill_mode": "fit"}
        },
        "detection_settings": {
            "tongue_sensitivity": 0.012,
            "eyebrow_frown_sensitivity": 0.003,
            "eyebrow_distance_threshold": 0.045,
            "smile_sensitivity": 0.008,
            "mouth_aspect_ratio_max": 0.2
        },
        "detection_mode": "face"  # face, hands, or both
    },
    "sillynubcat": {
        "directory": "sillynubcat",
        "description": "Sillynubcat expressions (hand gestures)",
        "expressions": {
            "1.jpeg": {"name": "Four Fingers", "fill_mode": "fit"},
            "2.png": {"name": "Index Finger", "fill_mode": "fit"},
            "3.jpeg": {"name": "Middle Finger", "fill_mode": "fit"}
        },
        "detection_settings": {
            "tongue_sensitivity": 0.015,
            "eyebrow_frown_sensitivity": 0.005,
            "eyebrow_distance_threshold": 0.050,
            "smile_sensitivity": 0.010,
            "mouth_aspect_ratio_max": 0.25
        },  
        "detection_mode": "hands"  # Changed to hand detection
    }
}

# Global variables
EXPRESSION_IMAGES = {}
EXPRESSION_SIZE = (400, 400)
BLACK_SQUARE = np.zeros((EXPRESSION_SIZE[1], EXPRESSION_SIZE[0], 3), dtype=np.uint8)
ACTIVE_CONFIG = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def select_folder_config():
    """Interactive folder selection"""
    print("\n" + "="*60)
    print("📁 Available Expression Folders:")
    print("="*60)
    
    configs = list(FOLDER_CONFIGS.items())
    for i, (key, config) in enumerate(configs, 1):
        print(f"{i}. {key.upper()}")
        print(f"   └─ {config['description']}")
        print(f"   └─ Mode: {config['detection_mode']}")
        print(f"   └─ Expressions: {len(config['expressions'])}")
    
    print("="*60)
    
    # Check command line argument
    if len(sys.argv) > 1:
        folder_name = sys.argv[1].lower()
        if folder_name in FOLDER_CONFIGS:
            print(f"✅ Using folder from argument: {folder_name}")
            return folder_name
        else:
            print(f"⚠️  Folder '{folder_name}' not found in configurations")
    
    # Interactive selection
    while True:
        try:
            choice = input(f"\nSelect folder (1-{len(configs)}) or enter name: ").strip()
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    return configs[idx][0]
            
            # Check if it's a folder name
            if choice.lower() in FOLDER_CONFIGS:
                return choice.lower()
            
            print("❌ Invalid selection. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Selection cancelled.")
            sys.exit(0)

def load_expression_image(img_path, fill_mode="fit"):
    """Load and resize an expression image based on fill mode"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    aspect = w / h
    target_aspect = EXPRESSION_SIZE[0] / EXPRESSION_SIZE[1]
    
    if fill_mode == "crop":
        # Fill the entire window (crop to fill)
        if aspect > target_aspect:
            new_h = EXPRESSION_SIZE[1]
            new_w = int(new_h * aspect)
        else:
            new_w = EXPRESSION_SIZE[0]
            new_h = int(new_w / aspect)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Crop center
        y_offset = (new_h - EXPRESSION_SIZE[1]) // 2
        x_offset = (new_w - EXPRESSION_SIZE[0]) // 2
        return img_resized[y_offset:y_offset+EXPRESSION_SIZE[1], 
                          x_offset:x_offset+EXPRESSION_SIZE[0]]
    else:
        # Fit mode: maintain aspect ratio with padding
        if aspect > 1:
            new_w = EXPRESSION_SIZE[0]
            new_h = int(new_w / aspect)
        else:
            new_h = EXPRESSION_SIZE[1]
            new_w = int(new_h * aspect)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.zeros((EXPRESSION_SIZE[1], EXPRESSION_SIZE[0], 3), dtype=np.uint8)
        y_offset = (EXPRESSION_SIZE[1] - new_h) // 2
        x_offset = (EXPRESSION_SIZE[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas

def load_folder_images(config):
    """Load all expression images for a given folder configuration"""
    global EXPRESSION_IMAGES
    EXPRESSION_IMAGES = {}
    
    folder_dir = config["directory"]
    expressions = config["expressions"]
    
    if not os.path.exists(folder_dir):
        print(f"⚠️  Warning: Directory '{folder_dir}' not found")
        return False
    
    print(f"\n📂 Loading images from '{folder_dir}'...")
    loaded_count = 0
    
    for img_file, expr_info in expressions.items():
        img_path = os.path.join(folder_dir, img_file)
        if os.path.exists(img_path):
            fill_mode = expr_info.get("fill_mode", "fit")
            img = load_expression_image(img_path, fill_mode)
            if img is not None:
                EXPRESSION_IMAGES[img_file] = img
                print(f"  ✓ {img_file} → {expr_info['name']} ({fill_mode})")
                loaded_count += 1
        else:
            print(f"  ✗ {img_file} not found")
    
    if loaded_count == 0:
        print(f"❌ No images loaded from '{folder_dir}'")
        return False
    
    print(f"✅ Loaded {loaded_count}/{len(expressions)} images\n")
    return True

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
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

# ============================================================================
# INITIALIZE CONFIGURATION
# ============================================================================

# Select and load folder configuration
selected_folder = select_folder_config()
ACTIVE_CONFIG = FOLDER_CONFIGS[selected_folder]

# Load images for selected folder
if not load_folder_images(ACTIVE_CONFIG):
    print("❌ Failed to load images. Exiting.")
    exit()

# Configure detectors based on detection mode
detection_mode = ACTIVE_CONFIG["detection_mode"]
face_mesh = None
hands = None

if detection_mode in ["face", "both"]:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ Face detection enabled")

if detection_mode in ["hands", "both"]:
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ Hand detection enabled")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print(f"\n✨ Face Tracker Started with '{selected_folder}' configuration!")
print(f"   Detection mode: {detection_mode.upper()}")
print("Press 'q' to quit, 's' to toggle styles")

# Visualization mode
viz_mode = 0  # 0: Full mesh, 1: Contours only, 2: Minimal

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_hand_gesture(hand_landmarks, config):
    """
    Detect hand gestures based on finger positions
    Returns: image filename or None
    """
    expressions = config["expressions"]
    landmarks = hand_landmarks.landmark
    
    # Finger tip and base landmark indices
    # Thumb: 4 (tip), 3, 2, 1 (base)
    # Index: 8 (tip), 7, 6, 5 (base)
    # Middle: 12 (tip), 11, 10, 9 (base)
    # Ring: 16 (tip), 15, 14, 13 (base)
    # Pinky: 20 (tip), 19, 18, 17 (base)
    # Wrist: 0
    
    def calculate_distance_3d(p1, p2):
        """Calculate 3D Euclidean distance between two landmarks"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_finger_extended(tip_idx, pip_idx, mcp_idx):
        """Check if a finger is extended by comparing distances"""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        wrist = landmarks[0]
        
        # Calculate distances from wrist
        tip_to_wrist = calculate_distance_3d(tip, wrist)
        pip_to_wrist = calculate_distance_3d(pip, wrist)
        
        # Also check if tip is above PIP joint (for vertical fingers)
        # In screen coordinates, smaller y = higher up
        tip_above_pip = tip.y < pip.y
        
        # Finger is extended if tip is further from wrist AND tip is above PIP
        return tip_to_wrist > pip_to_wrist or tip_above_pip
    
    def is_thumb_extended():
        """Check if thumb is extended"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]
        
        # Thumb is extended if tip is further from wrist than IP joint
        tip_dist = calculate_distance_3d(thumb_tip, wrist)
        ip_dist = calculate_distance_3d(thumb_ip, wrist)
        
        return tip_dist > ip_dist
    
    # Check each finger
    thumb_extended = is_thumb_extended()
    index_extended = is_finger_extended(8, 6, 5)
    middle_extended = is_finger_extended(12, 10, 9)
    ring_extended = is_finger_extended(16, 14, 13)
    pinky_extended = is_finger_extended(20, 18, 17)
    
    fingers_extended = [index_extended, middle_extended, ring_extended, pinky_extended]
    extended_count = sum(fingers_extended)
    
    # Debug print (optional - can be removed)
    # print(f"Thumb:{thumb_extended} Index:{index_extended} Middle:{middle_extended} Ring:{ring_extended} Pinky:{pinky_extended} Count:{extended_count}")
    
    # Gesture detection
    # 4 fingers up (not thumb) → 1.jpeg
    if extended_count == 4:
        gesture_files = [f for f in expressions.keys() if '1.' in f]
        if gesture_files:
            return gesture_files[0]
    
    # Only index finger up → 2.png
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        gesture_files = [f for f in expressions.keys() if '2.' in f]
        if gesture_files:
            return gesture_files[0]
    
    # Only middle finger up → 3.jpeg
    if middle_extended and not index_extended and not ring_extended and not pinky_extended:
        gesture_files = [f for f in expressions.keys() if '3.' in f]
        if gesture_files:
            return gesture_files[0]
    
    return None

def detect_expression(face_landmarks, config):
    """
    Detect facial expression based on landmark positions
    Uses configuration-specific detection settings
    Returns: image filename or None
    """
    # Key landmark indices
    landmarks = face_landmarks.landmark
    
    # Get detection settings from config
    settings = config["detection_settings"]
    expressions = config["expressions"]
    
    # Calculate mouth aspect ratio (height / width)
    mouth_height = calculate_distance(landmarks[13], landmarks[14])
    mouth_width = calculate_distance(landmarks[61], landmarks[291])
    mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
    
    # Calculate eye openness
    left_eye_height = calculate_distance(landmarks[159], landmarks[145])
    right_eye_height = calculate_distance(landmarks[386], landmarks[374])
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    
    # Detect tongue out (lower lip pushed down significantly)
    tongue_indicator = calculate_distance(landmarks[14], landmarks[17])
    
    # Calculate eyebrow position (for angry/frown detection)
    left_eyebrow_inner_y = landmarks[336].y
    right_eyebrow_inner_y = landmarks[107].y
    left_eye_top_y = landmarks[159].y
    right_eye_top_y = landmarks[386].y
    
    # Distance between inner eyebrows and eyes (smaller = furrowed)
    eyebrow_to_eye_left = left_eye_top_y - left_eyebrow_inner_y
    eyebrow_to_eye_right = right_eye_top_y - right_eyebrow_inner_y
    avg_eyebrow_distance = (eyebrow_to_eye_left + eyebrow_to_eye_right) / 2
    
    # Eyebrow angle/slant for angry (inner brows lower than outer)
    left_eyebrow_slant = left_eyebrow_inner_y - landmarks[285].y
    right_eyebrow_slant = right_eyebrow_inner_y - landmarks[55].y
    eyebrow_frown_angle = (left_eyebrow_slant + right_eyebrow_slant) / 2
    
    # Mouth corners position for smile detection
    left_corner_y = landmarks[61].y
    right_corner_y = landmarks[291].y
    mouth_corners_avg_y = (left_corner_y + right_corner_y) / 2
    upper_lip_y = landmarks[13].y
    lower_lip_y = landmarks[14].y
    mouth_center_y = (upper_lip_y + lower_lip_y) / 2
    
    # Smile: corners should be ABOVE the mouth center
    smile_indicator = mouth_center_y - mouth_corners_avg_y
    
    # Expression detection using config-specific thresholds
    # Check expressions in order based on what's available in config
    
    # Check for tongue out (typically 2nd expression)
    tongue_files = [f for f in expressions.keys() if '2.' in f]
    if tongue_files and tongue_indicator > settings["tongue_sensitivity"] and mouth_aspect_ratio > 0.05:
        return tongue_files[0]
    
    # Check for angry/furrowed brow (typically 1st expression)
    angry_files = [f for f in expressions.keys() if '1.' in f]
    if angry_files and (eyebrow_frown_angle > settings["eyebrow_frown_sensitivity"] or 
                        avg_eyebrow_distance < settings["eyebrow_distance_threshold"]):
        return angry_files[0]
    
    # Check for smile (typically 4th expression)
    smile_files = [f for f in expressions.keys() if '4.' in f]
    if smile_files and (smile_indicator > settings["smile_sensitivity"] and 
                        mouth_aspect_ratio < settings["mouth_aspect_ratio_max"]):
        return smile_files[0]
    
    # Check for shocked/mouth open (typically 3rd expression)
    shocked_files = [f for f in expressions.keys() if '3.' in f]
    # Add shocked detection if needed (currently disabled for nailong)
    
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
    
    # Detect expression and show corresponding image
    detected_expression = None
    
    # Process based on detection mode
    if detection_mode in ["face", "both"] and face_mesh:
        # Process face landmarks
        face_results = face_mesh.process(rgb_frame)
        
        # Draw face mesh annotations on the frame
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Detect expression
                detected_expression = detect_expression(face_landmarks, ACTIVE_CONFIG)
                
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
    
    # Process hand gestures
    if detection_mode in ["hands", "both"] and hands:
        hand_results = hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Detect hand gesture
                detected_gesture = detect_hand_gesture(hand_landmarks, ACTIVE_CONFIG)
                if detected_gesture:
                    detected_expression = detected_gesture
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
                )
    
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
    if detected_expression and detected_expression in ACTIVE_CONFIG["expressions"]:
        expression_name = ACTIVE_CONFIG["expressions"][detected_expression]["name"]
        expression_text = f"Expression: {expression_name}"
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
if face_mesh:
    face_mesh.close()
if hands:
    hands.close()
cv2.destroyAllWindows()
print("Face tracker closed.")

