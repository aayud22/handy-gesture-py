"""
config.py — Central configuration for HandyGesturePy.

All tunable constants live here so you can tweak behavior
without digging through multiple files.
"""

# ──────────────────────────────────────────────
#  MediaPipe Detection
# ──────────────────────────────────────────────
MAX_HANDS = 2                    # Track both hands
DETECTION_CONFIDENCE = 0.75      # Min confidence to initially detect a hand
TRACKING_CONFIDENCE = 0.75       # Min confidence to keep tracking between frames

# ──────────────────────────────────────────────
#  Mouse Smoothing (Exponential Moving Average)
# ──────────────────────────────────────────────
# Range: 0.0 → 1.0
#   Lower = smoother but more laggy
#   Higher = more responsive but jittery
SMOOTHING_FACTOR = 0.35

# ──────────────────────────────────────────────
#  Frame Reduction (Active Zone)
# ──────────────────────────────────────────────
# Pixels to shrink the usable camera area on each side.
# This lets you reach screen edges without stretching
# your hand to the very border of the camera frame.
FRAME_REDUCTION = 120

# ──────────────────────────────────────────────
#  Click (Pinch) Gesture
# ──────────────────────────────────────────────
CLICK_THRESHOLD = 30             # Max pixel distance between thumb & index to trigger a click
CLICK_COOLDOWN_SEC = 0.4         # Minimum seconds between consecutive clicks (debounce)

# ──────────────────────────────────────────────
#  Screenshot (Closed Left Fist Gesture)
# ──────────────────────────────────────────────
SCREENSHOT_COOLDOWN_SEC = 3.0    # 3-second debounce to prevent rapid screenshots
SCREENSHOTS_DIR = "screenshots"  # Folder to save screenshots (relative to project root)

# ──────────────────────────────────────────────
#  Air Paint (Right Hand Index-Only Gesture)
# ──────────────────────────────────────────────
AIR_PAINT_COLOR = (0, 255, 255)  # Yellow in BGR
AIR_PAINT_THICKNESS = 4          # Line thickness for painting strokes

# ──────────────────────────────────────────────
#  Two-Hand Fingertip Connection Lines
# ──────────────────────────────────────────────
# One color per finger pair: [Thumb, Index, Middle, Ring, Pinky]
FINGERTIP_COLORS = [
    (255, 0, 255),   # Magenta  — Thumb
    (0, 255, 0),     # Green    — Index
    (255, 255, 0),   # Cyan     — Middle
    (0, 165, 255),   # Orange   — Ring
    (0, 0, 255),     # Red      — Pinky
]
FINGERTIP_LINE_THICKNESS = 3

# ──────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────
WINDOW_TITLE = "HandyGesturePy"
