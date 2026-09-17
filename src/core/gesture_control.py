"""Gesture control module using MediaPipe and PyAutoGUI.

Detects hand landmarks and translates finger counts into system actions.
"""

import time
import mediapipe as mp
import pyautogui
import config


def count_fingers(hand_landmarks, frame_shape):
    """Count how many fingers are held up (0 to 5)."""
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark

    def xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    fingers = []

    # Thumb: tip x vs base joint x
    x4, y4 = xy(4)
    x2, y2 = xy(2)
    fingers.append(1 if x4 < x2 else 0)

    # 4 Fingers: tip y vs pip joint y (tip higher than joint => finger up)
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        xt, yt = xy(tip_id)
        xp, yp = xy(tip_id - 2)
        fingers.append(1 if yt < yp else 0)

    return sum(fingers)


class GestureController:
    """Manages hand tracking and action execution with cooldown."""

    def __init__(self, cooldown=config.GESTURE_COOLDOWN):
        self.cooldown = cooldown
        self.last_gesture_time = 0
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process_frame(self, rgb_frame, frame_shape, draw_target=None):
        """Process an RGB frame and execute corresponding gesture command.

        Returns (gesture_label, color, fingers_count, landmarks_detected)
        """
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return "No Hand", (200, 200, 200), None, False

        hand_lms = results.multi_hand_landmarks[0]
        if draw_target is not None:
            self.mp_draw.draw_landmarks(draw_target, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        fingers_up = count_fingers(hand_lms, frame_shape)
        now = time.time()

        if now - self.last_gesture_time < self.cooldown:
            return f"Cooldown ({fingers_up} fingers)", (0, 255, 255), fingers_up, True

        gesture_text = "Idle"
        gesture_color = (0, 255, 0)

        if fingers_up == 0:
            pyautogui.press("volumedown")
            gesture_text = "Volume Down"
            gesture_color = (0, 255, 255)
            self.last_gesture_time = now
        elif fingers_up == 1:
            pyautogui.press("volumeup")
            gesture_text = "Volume Up"
            gesture_color = (0, 255, 255)
            self.last_gesture_time = now
        elif fingers_up == 2:
            pyautogui.press("pagedown")
            gesture_text = "Next Slide"
            gesture_color = (0, 255, 255)
            self.last_gesture_time = now
        elif fingers_up == 3:
            pyautogui.press("pageup")
            gesture_text = "Previous Slide"
            gesture_color = (0, 255, 255)
            self.last_gesture_time = now
        elif fingers_up == 4:
            pyautogui.press("volumemute")
            gesture_text = "Mute / Unmute"
            gesture_color = (0, 255, 255)
            self.last_gesture_time = now
        else:
            gesture_text = "Hand Idle"
            gesture_color = (0, 255, 0)

        return gesture_text, gesture_color, fingers_up, True

    def close(self):
        self.hands.close()
