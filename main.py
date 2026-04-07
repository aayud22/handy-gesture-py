"""
main.py — HandyGesturePy entry point.

Pipeline (each frame):
  1. Capture frame from webcam & flip horizontally.
  2. Run hand detection via HandDetector.
  3. Extract index-finger and thumb landmarks.
  4. Map camera coordinates → screen coordinates (with frame reduction).
  5. Smooth the cursor position via MouseController (EMA).
  6. Check for pinch-to-click gesture (debounced).
  7. Draw visual feedback and FPS overlay.

Press 'q' to quit.
"""

import cv2
import numpy as np
import pyautogui
import time

from config import (
    CLICK_COOLDOWN_SEC,
    CLICK_THRESHOLD,
    DETECTION_CONFIDENCE,
    FRAME_REDUCTION,
    MAX_HANDS,
    SCREENSHOT_COOLDOWN_SEC,
    SCREENSHOTS_DIR,
    SMOOTHING_FACTOR,
    TRACKING_CONFIDENCE,
    WINDOW_TITLE,
)
from hand_detector import HandDetector
from mouse_controller import MouseController

# ──────────────────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────────────────

# Disable PyAutoGUI fail-safe (moving mouse to corner won't crash the app)
pyautogui.FAILSAFE = False

# Get the real screen resolution for coordinate mapping
screen_w, screen_h = pyautogui.size()

# Initialise modules
detector = HandDetector(
    max_hands=MAX_HANDS,
    detection_confidence=DETECTION_CONFIDENCE,
    tracking_confidence=TRACKING_CONFIDENCE,
)
controller = MouseController(
    screen_w=screen_w,
    screen_h=screen_h,
    smoothing_factor=SMOOTHING_FACTOR,
    click_threshold=CLICK_THRESHOLD,
    click_cooldown=CLICK_COOLDOWN_SEC,
    screenshot_cooldown=SCREENSHOT_COOLDOWN_SEC,
    screenshots_dir=SCREENSHOTS_DIR,
)

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check permissions in System Settings → Privacy → Camera.")
    exit(1)

# FPS tracking
prev_time = time.time()

print("✋ HandyGesturePy is running!")
print("   • Move your index finger to control the cursor.")
print("   • Pinch (thumb + index) to left-click.")
print("   • Close your LEFT fist to take a screenshot.")
print("   • Press 'q' in the camera window to quit.")


# ──────────────────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────────────────

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("WARNING: Failed to read frame, skipping...")
            continue

        # 1 — Mirror the frame so it feels natural (like looking in a mirror)
        frame = cv2.flip(frame, 1)
        cam_h, cam_w, _ = frame.shape

        # 2 — Run hand detection
        detector.detect(frame)

        if detector.hands_found():
            num_hands = detector.num_hands_detected()

            # ── Process EACH detected hand ────────────────
            for i in range(num_hands):
                landmarks = detector.get_landmark_positions(frame, hand_index=i)
                if len(landmarks) < 21:
                    continue

                index_tip = landmarks[8]   # (id, px_x, px_y)
                thumb_tip = landmarks[4]

                # Draw a label showing "Left" or "Right" near the wrist (LM 0)
                wrist = landmarks[0]
                hand_label = detector.get_handedness(hand_index=i)
                cv2.putText(
                    frame,
                    hand_label,
                    (wrist[1] - 20, wrist[2] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # ── Mouse control: use the FIRST hand only ──
                if i == 0:
                    # 4 — Map camera coords → screen coords
                    mapped_x = np.interp(
                        index_tip[1],
                        [FRAME_REDUCTION, cam_w - FRAME_REDUCTION],
                        [0, screen_w],
                    )
                    mapped_y = np.interp(
                        index_tip[2],
                        [FRAME_REDUCTION, cam_h - FRAME_REDUCTION],
                        [0, screen_h],
                    )

                    # 5 — Smooth & move the mouse cursor
                    controller.smooth_move(mapped_x, mapped_y)

                    # 6 — Check for pinch-to-click gesture
                    clicked = controller.check_click(thumb_tip, index_tip)

                    # Highlight the controlling hand's fingertips
                    color = (0, 255, 0) if clicked else (0, 0, 255)
                    cv2.circle(frame, (index_tip[1], index_tip[2]), 12, color, cv2.FILLED)
                    cv2.circle(frame, (thumb_tip[1], thumb_tip[2]), 12, color, cv2.FILLED)

                    # Draw pinch distance line
                    distance = controller.distance(thumb_tip, index_tip)
                    line_color = (0, 255, 0) if distance < CLICK_THRESHOLD else (255, 200, 0)
                    cv2.line(
                        frame,
                        (thumb_tip[1], thumb_tip[2]),
                        (index_tip[1], index_tip[2]),
                        line_color,
                        2,
                    )
                else:
                    # Non-controlling hands: draw fingertips in a neutral color
                    cv2.circle(frame, (index_tip[1], index_tip[2]), 8, (255, 200, 0), cv2.FILLED)
                    cv2.circle(frame, (thumb_tip[1], thumb_tip[2]), 8, (255, 200, 0), cv2.FILLED)

                # ── Screenshot: closed LEFT fist ──────────
                hand_label = detector.get_handedness(hand_index=i)
                if hand_label == "Left":
                    if detector.is_fist(landmarks):
                        took_screenshot = controller.take_screenshot()
                        if took_screenshot:
                            # Flash a purple banner on the frame
                            cv2.putText(
                                frame,
                                "SCREENSHOT!",
                                (cam_w // 2 - 120, cam_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255, 0, 255),
                                3,
                            )

            # Draw skeletons (landmarks + connections) for ALL hands
            detector.draw_all(frame)

        # ── FPS overlay ───────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 100),
            2,
        )

        # Show the frame
        cv2.imshow(WINDOW_TITLE, frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # ── Cleanup ───────────────────────────────────────────
    print("\n🛑 Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Resources released. Goodbye!")