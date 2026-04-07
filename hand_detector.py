"""
hand_detector.py — Encapsulates MediaPipe Hand Landmarker detection.

Uses the modern mediapipe.tasks API (mp.tasks.vision.HandLandmarker)
which requires a downloaded .task model bundle.

Responsibilities:
  • Initialise and manage the HandLandmarker model.
  • Accept a BGR frame, convert to mp.Image, and run detection.
  • Extract landmark pixel positions from results.
  • Determine which fingers are extended (up) or curled (down).
  • Detect closed-fist gesture for screenshot trigger.
  • Draw landmarks + connections on the frame.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision, BaseOptions


# Path to the model bundle (relative to this file)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_DEFAULT_MODEL = os.path.join(_MODEL_DIR, "hand_landmarker.task")


class HandDetector:
    """Wrapper around MediaPipe HandLandmarker (tasks API) for clean,
    reusable hand detection."""

    # MediaPipe landmark indices for quick reference:
    #   WRIST=0, THUMB_TIP=4, INDEX_TIP=8, MIDDLE_TIP=12,
    #   RING_TIP=16, PINKY_TIP=20
    # Full chart: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_hands: int = 2,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}.\n"
                "Download it with:\n"
                "  curl --http1.1 -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        # Configure in VIDEO mode (synchronous, frame-by-frame processing)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Drawing utilities from the tasks API
        self._draw = vision.drawing_utils
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        # Cache for the latest detection results
        self._results = None

        # Monotonically increasing timestamp for VIDEO mode (milliseconds)
        self._timestamp_ms: int = 0

    # ══════════════════════════════════════════════════════
    #  CORE DETECTION
    # ══════════════════════════════════════════════════════

    def detect(self, frame):
        """Run hand detection on a BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            HandLandmarkerResult (also cached internally).
        """
        # Convert BGR → RGB and wrap in MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps MUST be strictly increasing in VIDEO mode
        self._timestamp_ms += 33  # ~30 fps increment
        self._results = self._landmarker.detect_for_video(
            mp_image, self._timestamp_ms
        )
        return self._results

    def hands_found(self) -> bool:
        """Return True if the last detect() call found at least one hand."""
        return (
            self._results is not None
            and len(self._results.hand_landmarks) > 0
        )

    def num_hands_detected(self) -> int:
        """Return the number of hands detected in the last frame."""
        if not self.hands_found():
            return 0
        return len(self._results.hand_landmarks)

    # ══════════════════════════════════════════════════════
    #  HANDEDNESS
    # ══════════════════════════════════════════════════════

    def get_handedness(self, hand_index: int = 0) -> str:
        """Return 'Left' or 'Right' for the given hand.

        IMPORTANT: Since we mirror the frame with cv2.flip(), MediaPipe's
        labels are inverted — it sees your left hand as 'Right' and vice
        versa. We swap them here so the label matches the USER's actual hand.
        """
        if not self.hands_found() or hand_index >= len(self._results.handedness):
            return "Unknown"

        raw_label = self._results.handedness[hand_index][0].category_name
        # Swap because the frame is horizontally mirrored
        return "Left" if raw_label == "Right" else "Right"

    # ══════════════════════════════════════════════════════
    #  LANDMARK POSITIONS
    # ══════════════════════════════════════════════════════

    def get_landmark_positions(self, frame, hand_index: int = 0) -> list:
        """Convert normalised landmarks → pixel coordinates.

        Args:
            frame:      The original camera frame (used for its dimensions).
            hand_index: Which detected hand to query (default 0 = first).

        Returns:
            List of tuples: [(id, pixel_x, pixel_y), ...] for 21 landmarks.
            Empty list if no hand is available.
        """
        if not self.hands_found():
            return []
        if hand_index >= len(self._results.hand_landmarks):
            return []

        h, w, _ = frame.shape
        hand_landmarks = self._results.hand_landmarks[hand_index]

        positions = []
        for lm_id, lm in enumerate(hand_landmarks):
            px_x = int(lm.x * w)
            px_y = int(lm.y * h)
            positions.append((lm_id, px_x, px_y))

        return positions

    # ══════════════════════════════════════════════════════
    #  GESTURE STATE DETECTION
    # ══════════════════════════════════════════════════════

    def fingers_up(self, landmarks: list, handedness: str = "Right") -> list:
        """Determine which of the 5 fingers are extended (up/open).

        Logic for each finger:
          • Thumb:  Uses X-axis comparison because the thumb moves
                    laterally. Direction depends on which hand it is.
                    In the MIRRORED frame:
                      - User's Left hand → thumb extends to the RIGHT
                        → extended if tip.x > IP.x
                      - User's Right hand → thumb extends to the LEFT
                        → extended if tip.x < IP.x
          • Index / Middle / Ring / Pinky:
                    Uses Y-axis comparison. Fingertip ABOVE its PIP
                    joint means the finger is extended.
                    (Y increases downward, so tip.y < PIP.y → extended)

        Args:
            landmarks:  List of (id, px_x, px_y) from get_landmark_positions().
            handedness: 'Left' or 'Right' (from get_handedness()).

        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky].
            Returns [False]*5 if landmarks are insufficient.
        """
        if len(landmarks) < 21:
            return [False] * 5

        fingers = []

        # ── Thumb (horizontal check) ──
        # Landmark 4 = thumb tip, Landmark 3 = thumb IP joint
        thumb_tip_x = landmarks[4][1]
        thumb_ip_x = landmarks[3][1]

        if handedness == "Left":
            # User's left hand in mirrored frame: thumb points RIGHT
            fingers.append(thumb_tip_x > thumb_ip_x)
        else:
            # User's right hand in mirrored frame: thumb points LEFT
            fingers.append(thumb_tip_x < thumb_ip_x)

        # ── Index, Middle, Ring, Pinky (vertical check) ──
        # Pairs: (fingertip_landmark, PIP_joint_landmark)
        for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            # Tip ABOVE PIP joint → finger is extended (smaller y = higher)
            fingers.append(landmarks[tip_id][2] < landmarks[pip_id][2])

        return fingers

    def is_fist(self, landmarks: list) -> bool:
        """Detect if the hand is making a closed fist (all 4 fingers curled).

        Checks index, middle, ring, pinky: each fingertip y must be
        BELOW (greater than) its PIP joint y. Thumb is excluded because
        its lateral movement makes it unreliable for fist detection.

        Args:
            landmarks: List of (id, px_x, px_y) from get_landmark_positions().

        Returns:
            True if the hand is a closed fist.
        """
        if len(landmarks) < 21:
            return False

        # Fingertip vs PIP joint pairs
        # Tip:  8, 12, 16, 20     (index, middle, ring, pinky)
        # PIP:  6, 10, 14, 18
        for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip_id][2] < landmarks[pip_id][2]:
                # Fingertip is ABOVE the PIP joint → finger is still extended
                return False

        return True

    # ══════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════

    def draw(self, frame, hand_index: int = 0):
        """Draw landmarks and hand connections on the frame.

        Args:
            frame:      The BGR frame to draw on (modified in-place).
            hand_index: Which detected hand to draw.
        """
        if not self.hands_found():
            return
        if hand_index >= len(self._results.hand_landmarks):
            return

        hand_landmarks = self._results.hand_landmarks[hand_index]

        # The tasks API draw_landmarks expects the normalised landmark list
        self._draw.draw_landmarks(
            frame,
            hand_landmarks,
            self._hand_connections,
        )

    def draw_all(self, frame):
        """Draw landmarks and skeleton connections for ALL detected hands.

        Args:
            frame: The BGR frame to draw on (modified in-place).
        """
        for i in range(self.num_hands_detected()):
            self.draw(frame, hand_index=i)

    # ══════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
