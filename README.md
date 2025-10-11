# face-tracker

Real-time facial expression tracking with beautiful overlays.

## Installation

```bash
pip install -r requirements.txt
```

**Note for Mac M1/M2:** The requirements file automatically installs `mediapipe-silicon` for better ARM compatibility.

## Usage

```bash
python face_tracker.py
```

**Controls:**
- Press `S` to cycle through visualization modes
- Press `Q` to quit

## Features

- Real-time face mesh with 468 facial landmarks
- **Expression-based image mapping** - images appear based on your expressions:
  - Neutral: Black square
  - **Angry** (frown eyebrows): `1.jpeg`
  - **Tongue Out**: `2.jpg`
  - **Shocked** (wide open mouth + eyes): `3.jpg`
  - **Smiling** (corners up, closed mouth): `4.png`
- Three visualization modes:
  - **FULL MESH**: Complete 3D face mesh with all landmarks
  - **CONTOURS**: Clean face and eye contours
  - **MINIMAL**: Aesthetic mode with eyes, eyebrows, and lips
- Transparent white overlay (25% opacity)
- Mirrored view for natural interaction
- Expression images shown in separate window at consistent size