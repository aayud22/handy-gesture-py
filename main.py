"""
main.py — HandyGesturePy entry point.

Pipeline (each frame):
  1. Capture frame from webcam & flip horizontally (mirror).
  2. Run hand detection via HandDetector.
  3. Classify each detected hand as Left or Right.
  4. Left hand:  closed fist → take screenshot (3 s debounce).
  5. Right hand: gesture determines mode:
       • Only index finger up       → AIR PAINT mode
       • Index + middle fingers up  → MOUSE CONTROL mode (+ pinch to click)
       • All 5 fingers open         → CLEAR the paint canvas
  6. Both hands visible: draw coloured connection lines between
     corresponding fingertips (thumb↔thumb, index↔index, etc.).
  7. Overlay the air-paint canvas, draw skeletons, FPS, and mode label.

Press 'q' to quit.
"""

import cv2
import numpy as np
import pyautogui
import time

from config import (
    AIR_PAINT_COLOR,
    AIR_PAINT_THICKNESS,
    CLICK_COOLDOWN_SEC,
    CLICK_THRESHOLD,
    DETECTION_CONFIDENCE,
    FINGERTIP_COLORS,
    FINGERTIP_LINE_THICKNESS,
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


# ══════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════

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
    print("ERROR: Cannot open webcam.")
    print("       Check System Settings → Privacy & Security → Camera.")
    exit(1)

# ── Air Paint State ────────────────────────────────────────
#  air_canvas: a blank black image the same size as the camera frame.
#              Paint strokes are drawn here, then overlaid on the video.
#  prev_paint_point: the last (x, y) where we drew, so we can draw
#                    continuous lines. None = "pen lifted" (no line to here).
air_canvas = None           # Created on first frame (need frame dimensions)
prev_paint_point = None     # (x, y) or None

# ── FPS tracking ──────────────────────────────────────────
prev_time = time.time()

# ── Screenshot visual feedback timer ──────────────────────
#  When a screenshot is taken, we show a banner for ~1 second.
last_screenshot_time = 0.0

print("✋ HandyGesturePy is running!")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  RIGHT HAND gestures:")
print("    ☝️  Index only       → Air Paint")
print("    ✌️  Index + Middle   → Mouse Control")
print("    👋 All 5 fingers    → Clear Canvas")
print("    🤏 Pinch            → Left Click (in mouse mode)")
print("")
print("  LEFT HAND gestures:")
print("    ✊ Closed fist      → Screenshot (3 s cooldown)")
print("")
print("  BOTH HANDS:")
print("    🤝 Fingertip connection lines drawn automatically")
print("")
print("  Press 'q' in the camera window to quit.")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # ── 1. Mirror the frame (feels natural, like a mirror) ──
        frame = cv2.flip(frame, 1)
        cam_h, cam_w, _ = frame.shape

        # ── Initialise air_canvas on the first frame ──
        #    (We need frame dims which aren't known until now.)
        if air_canvas is None:
            air_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        # ── 2. Run hand detection ──
        detector.detect(frame)

        # ── 3. Classify hands as Left / Right ──
        #    We iterate once to sort hands by handedness so that
        #    later logic can act on a specific hand independently.
        left_landmarks = None
        right_landmarks = None
        left_idx = None
        right_idx = None

        if detector.hands_found():
            for i in range(detector.num_hands_detected()):
                lm = detector.get_landmark_positions(frame, hand_index=i)
                if len(lm) < 21:
                    continue
                handedness = detector.get_handedness(hand_index=i)
                if handedness == "Left":
                    left_landmarks = lm
                    left_idx = i
                elif handedness == "Right":
                    right_landmarks = lm
                    right_idx = i

        # ══════════════════════════════════════════════════
        #  4. LEFT HAND — SCREENSHOT ON CLOSED FIST
        # ══════════════════════════════════════════════════
        #  Detect if ALL 4 fingers are curled (fist). If so,
        #  fire a debounced screenshot via the controller.

        if left_landmarks is not None:
            # Draw "Left" label near the wrist (Landmark 0)
            wrist = left_landmarks[0]
            cv2.putText(
                frame, "Left",
                (wrist[1] - 30, wrist[2] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            # Check for fist gesture
            if detector.is_fist(left_landmarks):
                took_screenshot = controller.take_screenshot()
                if took_screenshot:
                    last_screenshot_time = time.time()

        # Show "SCREENSHOT!" banner for ~1 second after capturing
        if (time.time() - last_screenshot_time) < 1.0:
            cv2.putText(
                frame, "SCREENSHOT SAVED!",
                (cam_w // 2 - 180, cam_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3,
            )

        # ══════════════════════════════════════════════════
        #  5. RIGHT HAND — GESTURE MODE SELECTION
        # ══════════════════════════════════════════════════
        #  Determine the current gesture mode based on which
        #  fingers are extended.

        right_mode = "IDLE"

        if right_landmarks is not None:
            # Get the boolean list: [thumb, index, middle, ring, pinky]
            fingers = detector.fingers_up(right_landmarks, "Right")

            # Draw "Right" label near the wrist
            wrist = right_landmarks[0]
            # (label text updated below with mode)

            # ─────────────────────────────────────────────
            #  MODE A: AIR PAINT — only index finger extended
            # ─────────────────────────────────────────────
            #  Condition: index=True, middle/ring/pinky=False.
            #  (Thumb state is ignored — it's hard to control precisely.)
            if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                right_mode = "PAINT"

                # Get the index fingertip position (Landmark 8)
                cx, cy = right_landmarks[8][1], right_landmarks[8][2]

                # Draw a continuous line on the air canvas
                if prev_paint_point is not None:
                    cv2.line(
                        air_canvas,
                        prev_paint_point,
                        (cx, cy),
                        AIR_PAINT_COLOR,
                        AIR_PAINT_THICKNESS,
                    )
                # Update the "pen" position for the next frame
                prev_paint_point = (cx, cy)

                # Visual feedback: bright circle on the fingertip
                cv2.circle(frame, (cx, cy), 10, AIR_PAINT_COLOR, cv2.FILLED)

            # ─────────────────────────────────────────────
            #  MODE B: MOUSE CONTROL — index + middle extended
            # ─────────────────────────────────────────────
            #  Condition: index=True, middle=True, ring/pinky=False.
            #  The cursor follows the index fingertip.
            #  Pinching thumb+index triggers a click.
            elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                right_mode = "MOUSE"
                prev_paint_point = None  # Lift the "pen"

                index_tip = right_landmarks[8]
                thumb_tip = right_landmarks[4]

                # Map camera coordinates → screen coordinates
                # np.interp with FRAME_REDUCTION creates a smaller
                # "active zone" so you don't have to reach frame edges.
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

                # Smooth & move the OS cursor
                controller.smooth_move(mapped_x, mapped_y)

                # Check for pinch-to-click
                clicked = controller.check_click(thumb_tip, index_tip)

                # Visual feedback: red circle → green on click
                color = (0, 255, 0) if clicked else (0, 0, 255)
                cv2.circle(frame, (index_tip[1], index_tip[2]), 12, color, cv2.FILLED)
                cv2.circle(frame, (thumb_tip[1], thumb_tip[2]), 12, color, cv2.FILLED)

                # Draw a line between thumb and index showing pinch distance
                dist = controller.distance(thumb_tip, index_tip)
                line_col = (0, 255, 0) if dist < CLICK_THRESHOLD else (255, 200, 0)
                cv2.line(
                    frame,
                    (thumb_tip[1], thumb_tip[2]),
                    (index_tip[1], index_tip[2]),
                    line_col, 2,
                )

            # ─────────────────────────────────────────────
            #  MODE C: CLEAR CANVAS — all 5 fingers extended
            # ─────────────────────────────────────────────
            elif all(fingers):
                right_mode = "CLEAR"
                prev_paint_point = None  # Lift the "pen"

                # Wipe the air canvas clean
                air_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

                # Visual feedback
                cv2.putText(
                    frame, "CANVAS CLEARED",
                    (cam_w // 2 - 140, cam_h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2,
                )

            # ─────────────────────────────────────────────
            #  ANY OTHER GESTURE → idle, reset paint pen
            # ─────────────────────────────────────────────
            else:
                prev_paint_point = None  # Lift the "pen"

            # Draw the mode label near the right wrist
            cv2.putText(
                frame, f"Right [{right_mode}]",
                (wrist[1] - 40, wrist[2] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        else:
            # No right hand detected → lift the pen
            prev_paint_point = None

        # ══════════════════════════════════════════════════
        #  6. BOTH HANDS — FINGERTIP CONNECTION LINES
        # ══════════════════════════════════════════════════
        #  When exactly 2 hands are detected, draw coloured
        #  lines between corresponding fingertips.
        #  Landmark IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20

        if left_landmarks is not None and right_landmarks is not None:
            fingertip_ids = [4, 8, 12, 16, 20]

            for finger_idx, tip_id in enumerate(fingertip_ids):
                # Get pixel positions for left and right fingertips
                pt_left = (left_landmarks[tip_id][1], left_landmarks[tip_id][2])
                pt_right = (right_landmarks[tip_id][1], right_landmarks[tip_id][2])

                # Pick the colour for this finger pair
                color = FINGERTIP_COLORS[finger_idx]

                # Draw the connecting line
                cv2.line(frame, pt_left, pt_right, color, FINGERTIP_LINE_THICKNESS)

                # Draw circles at connection endpoints for visibility
                cv2.circle(frame, pt_left, 7, color, cv2.FILLED)
                cv2.circle(frame, pt_right, 7, color, cv2.FILLED)

        # ══════════════════════════════════════════════════
        #  7. DRAW HAND SKELETONS
        # ══════════════════════════════════════════════════

        if detector.hands_found():
            detector.draw_all(frame)

        # ══════════════════════════════════════════════════
        #  8. OVERLAY AIR CANVAS ONTO VIDEO FRAME
        # ══════════════════════════════════════════════════
        #  Wherever the canvas has colour (non-black pixels),
        #  replace the video pixels with the paint colour.
        #  This creates a clean overlay without affecting the
        #  rest of the video feed.

        if air_canvas is not None:
            # Create a mask of painted pixels
            gray_canvas = cv2.cvtColor(air_canvas, cv2.COLOR_BGR2GRAY)
            _, paint_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)

            # Inverse mask: where there is NO paint
            mask_inv = cv2.bitwise_not(paint_mask)

            # Black out the painted regions in the video frame
            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            # Isolate just the paint strokes
            fg = cv2.bitwise_and(air_canvas, air_canvas, mask=paint_mask)
            # Combine: clean video + paint overlay
            frame = cv2.add(bg, fg)

        # ══════════════════════════════════════════════════
        #  9. HUD — FPS + MODE OVERLAY
        # ══════════════════════════════════════════════════

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # FPS counter (top-left)
        cv2.putText(
            frame, f"FPS: {int(fps)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2,
        )

        # Active mode indicator (top-right)
        cv2.putText(
            frame, f"Mode: {right_mode}",
            (cam_w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

        # ── Display ───────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # ══════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════
    print("\n🛑 Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Resources released. Goodbye!")