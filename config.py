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
#  Screenshot (Closed Left Palm Gesture)
# ──────────────────────────────────────────────
SCREENSHOT_COOLDOWN_SEC = 1.5    # Minimum seconds between screenshots (debounce)
SCREENSHOTS_DIR = "screenshots"  # Folder to save screenshots (relative to project root)

# ──────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────
WINDOW_TITLE = "HandyGesturePy — Mouse Control"
