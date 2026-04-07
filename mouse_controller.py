"""
mouse_controller.py — Handles mouse movement smoothing and click detection.

Responsibilities:
  • Smooth raw coordinates via Exponential Moving Average (EMA).
  • Map camera-space coordinates to screen-space.
  • Detect pinch gesture (thumb ↔ index) and fire a debounced click.
  • Take debounced screenshots on command.
"""

import math
import os
import time
from datetime import datetime

import pyautogui


class MouseController:
    """Smooth mouse control with pinch-to-click, debounce, and screenshot."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        smoothing_factor: float = 0.35,
        click_threshold: float = 30,
        click_cooldown: float = 0.4,
        screenshot_cooldown: float = 1.5,
        screenshots_dir: str = "screenshots",
    ):
        """
        Args:
            screen_w:            Screen width in pixels (from pyautogui.size()).
            screen_h:            Screen height in pixels.
            smoothing_factor:    EMA weight — 0 = max smooth, 1 = raw/no smooth.
            click_threshold:     Max pixel distance between thumb & index to count
                                 as a "pinch" (click trigger).
            click_cooldown:      Minimum seconds between consecutive clicks.
            screenshot_cooldown: Minimum seconds between consecutive screenshots.
            screenshots_dir:     Folder path to save screenshot images.
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.smoothing = smoothing_factor
        self.click_threshold = click_threshold
        self.click_cooldown = click_cooldown
        self.screenshot_cooldown = screenshot_cooldown
        self.screenshots_dir = screenshots_dir

        # EMA state — previous smoothed position
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._initialized: bool = False

        # Click debounce state
        self._last_click_time: float = 0.0

        # Screenshot debounce state
        self._last_screenshot_time: float = 0.0

    # ── Mouse Movement ────────────────────────────────────

    def smooth_move(self, raw_x: float, raw_y: float):
        """Apply EMA smoothing and move the OS cursor.

        The first frame is used to seed the EMA so the cursor doesn't
        jump from (0, 0).

        Args:
            raw_x: Mapped screen-space x (after np.interp).
            raw_y: Mapped screen-space y.
        """
        if not self._initialized:
            # Seed with the first real position to avoid a jump from origin
            self._prev_x = raw_x
            self._prev_y = raw_y
            self._initialized = True

        # Exponential Moving Average
        curr_x = self._prev_x + self.smoothing * (raw_x - self._prev_x)
        curr_y = self._prev_y + self.smoothing * (raw_y - self._prev_y)

        # Clamp to screen bounds
        curr_x = max(0, min(self.screen_w - 1, curr_x))
        curr_y = max(0, min(self.screen_h - 1, curr_y))

        # Move the mouse — _pause=0 avoids PyAutoGUI's default 0.1s delay
        pyautogui.moveTo(int(curr_x), int(curr_y), _pause=False)

        # Update state for next frame
        self._prev_x = curr_x
        self._prev_y = curr_y

    # ── Click Detection ───────────────────────────────────

    def check_click(
        self, thumb_pos: tuple, index_pos: tuple
    ) -> bool:
        """Check for a pinch gesture and fire a debounced click.

        Args:
            thumb_pos: (id, px_x, px_y) of the thumb tip (Landmark 4).
            index_pos: (id, px_x, px_y) of the index tip (Landmark 8).

        Returns:
            True if a click was triggered this frame, False otherwise.
        """
        # Euclidean distance between thumb tip and index finger tip
        distance = math.hypot(
            thumb_pos[1] - index_pos[1],  # dx
            thumb_pos[2] - index_pos[2],  # dy
        )

        now = time.time()
        if distance < self.click_threshold:
            if (now - self._last_click_time) > self.click_cooldown:
                pyautogui.click(_pause=False)
                self._last_click_time = now
                return True

        return False

    # ── Screenshot ────────────────────────────────────────

    def take_screenshot(self) -> bool:
        """Take a debounced screenshot and save it to the screenshots folder.

        Returns:
            True if a screenshot was taken, False if cooldown is still active.
        """
        now = time.time()
        if (now - self._last_screenshot_time) < self.screenshot_cooldown:
            return False

        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.screenshots_dir, f"screenshot_{timestamp}.png")

        # Capture the screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)

        self._last_screenshot_time = now
        print(f"📸 Screenshot saved: {filepath}")
        return True

    # ── Utilities ─────────────────────────────────────────

    @staticmethod
    def distance(pos_a: tuple, pos_b: tuple) -> float:
        """Utility: Euclidean distance between two landmark tuples."""
        return math.hypot(pos_a[1] - pos_b[1], pos_a[2] - pos_b[2])
