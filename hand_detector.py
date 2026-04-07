"""
hand_detector.py — Encapsulates MediaPipe Hand Landmarker detection.

Uses the modern mediapipe.tasks API (mp.tasks.vision.HandLandmarker)
which requires a downloaded .task model bundle.

Responsibilities:
  • Initialise and manage the HandLandmarker model.
  • Accept a BGR frame, convert to mp.Image, and run detection.
  • Extract landmark pixel positions from results.
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

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}.\n"
                "Download it with:\n"
                "  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        # Configure in VIDEO mode (synchronous, frame-by-frame)
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

        # Cache for the latest results
        self._results = None

        # Monotonically increasing timestamp for VIDEO mode (milliseconds)
        self._timestamp_ms: int = 0

    # ── Public API ────────────────────────────────────────

    def detect(self, frame):
        """Run hand detection on a BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            HandLandmarkerResult (also cached internally).
        """
        # Convert BGR → RGB and wrap in mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly increasing in VIDEO mode
        self._timestamp_ms += 33  # ~30 fps
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

    def get_handedness(self, hand_index: int = 0) -> str:
        """Return 'Left' or 'Right' for the given hand.

        Since we mirror the frame with cv2.flip(), MediaPipe's labels
        are inverted — it sees your left hand as 'Right' and vice versa.
        We swap them here so the label matches the USER's actual hand.
        """
        if not self.hands_found() or hand_index >= len(self._results.handedness):
            return "Unknown"

        raw_label = self._results.handedness[hand_index][0].category_name
        # Swap because the frame is horizontally mirrored
        return "Left" if raw_label == "Right" else "Right"

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

        # The tasks API draw_landmarks expects normalized landmarks list
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

    def is_fist(self, landmarks: list) -> bool:
        """Detect if the hand is making a closed fist (all fingers curled in).

        Checks the 4 main fingers (index, middle, ring, pinky):
          • Each fingertip y must be BELOW its PIP joint y
            (in pixel coords y increases downward, so curled = larger y)

        The thumb is intentionally excluded from the check because
        its direction depends on handedness and orientation, making
        it unreliable. A closed fist is well-detected by just the
        4 fingers.

        Args:
            landmarks: List of (id, px_x, px_y) from get_landmark_positions().

        Returns:
            True if the hand is a closed fist.
        """
        if len(landmarks) < 21:
            return False

        # Fingertip and PIP joint landmark indices
        # Tip:  8, 12, 16, 20
        # PIP:  6, 10, 14, 18
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        # All 4 fingers must be curled (tip y > PIP y)
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if landmarks[tip_id][2] < landmarks[pip_id][2]:
                # Fingertip is ABOVE the PIP joint → finger is extended
                return False

        return True

    # ── Cleanup ───────────────────────────────────────────

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
