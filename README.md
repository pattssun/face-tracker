# face-tracker

Real-time facial expression tracking with beautiful overlays.

## Requirements

- Python 3.9, 3.10, or 3.11 (MediaPipe does not support 3.13 yet)

## Installation

```bash
# Create virtual environment with compatible Python version
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** If you're on Mac M1/M2 and encounter issues, try `pip install mediapipe-silicon` instead of mediapipe.

## Usage

### Interactive Mode (Default)
```bash
python face_tracker.py
```
You'll be prompted to select from available expression folders.

### Command-Line Mode
```bash
# Use a specific folder directly
python face_tracker.py nailong
python face_tracker.py sillynubcat
```

**Controls:**
- Press `S` to cycle through visualization modes
- Press `Q` to quit

## Features

- Real-time face mesh with 468 facial landmarks
- **Modular folder-based configuration system**
  - Each folder has its own expression mappings and detection settings
  - Easy to add new expression sets
  - Customizable detection sensitivity per folder
- **Expression-based image mapping** - images appear based on your expressions:
  - Neutral: Black square
  - Expression detection includes: Angry, Tongue Out, Shocked, Smiling
  - Auto-detects and loads available expressions from selected folder
- Three visualization modes:
  - **FULL MESH**: Complete 3D face mesh with all landmarks
  - **CONTOURS**: Clean face and eye contours
  - **MINIMAL**: Aesthetic mode with eyes, eyebrows, and lips
- Transparent white overlay (25% opacity)
- Mirrored view for natural interaction
- Expression images shown in separate window at consistent size

## Detection Modes

### Face Detection (nailong)
- Detects facial expressions using MediaPipe Face Mesh
- Maps expressions like: Angry, Tongue Out, Shocked, Smiling

### Hand Gesture Detection (sillynubcat)
- Detects hand gestures using MediaPipe Hands
- Supported gestures:
  - **4 Fingers**: All fingers up (no thumb) → `1.jpeg`
  - **Index Finger**: Only index finger pointing up → `2.png`
  - **Middle Finger**: Only middle finger up → `3.jpeg`

### Both Modes
- Set `"detection_mode": "both"` to use face and hand detection together

## Adding New Expression Folders

1. Create a new folder with your expression images (e.g., `myexpressions/`)
2. Add numbered images: `1.jpeg`, `2.jpeg`, `3.png`, `4.jpeg`, etc.
3. Add configuration in `face_tracker.py`:

```python
FOLDER_CONFIGS = {
    # ... existing configs ...
    "myexpressions": {
        "directory": "myexpressions",
        "description": "My custom expressions",
        "expressions": {
            "1.jpeg": {"name": "Happy", "fill_mode": "fit"},
            "2.jpeg": {"name": "Sad", "fill_mode": "crop"},
            # Add more...
        },
        "detection_settings": {
            "tongue_sensitivity": 0.012,
            "eyebrow_frown_sensitivity": 0.003,
            "eyebrow_distance_threshold": 0.045,
            "smile_sensitivity": 0.008,
            "mouth_aspect_ratio_max": 0.2
        },
        "detection_mode": "face"  # Options: "face", "hands", or "both"
    }
}
```

4. Run with: `python face_tracker.py myexpressions`