import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

ACTION_COOLDOWN = 0.5 


def count_fingers(hand_landmarks, frame_shape):
    """
    Returns how many fingers are up (0-5)
    """
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark

    def xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    fingers = []

    x4, y4 = xy(4)   # thumb tip
    x2, y2 = xy(2)   # thumb joint
    fingers.append(1 if x4 < x2 else 0)

    # Other 4 fingers: tip above pip joint => finger up
    tip_ids = [8, 12, 16, 20]  
    for tip_id in tip_ids:
        xt, yt = xy(tip_id)
        xp, yp = xy(tip_id - 2)  # pip joint
        fingers.append(1 if yt < yp else 0)

    return sum(fingers)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    last_action_time = 0
    print("[INFO] Gesture slide control running. Press Q to quit.")
    print("Hold 1 finger up = Previous slide")
    print("Hold 2 fingers up = Next slide")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        action_text = "No action"

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # draw hand skeleton
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                fingers_up = count_fingers(handLms, frame.shape)

                cv2.putText(frame, f"Fingers: {fingers_up}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                now = time.time()
                time_gap = now - last_action_time

                if time_gap < ACTION_COOLDOWN:
                    action_text = f"Cooldown ({time_gap:.2f}s)"
                else:
                    if fingers_up == 1:
                        pyautogui.press('left')
                        action_text = "Previous slide"
                        last_action_time = now
                        print("[ACTION] Previous slide")
                    elif fingers_up == 2:
                        pyautogui.press('right')
                        action_text = "Next slide"
                        last_action_time = now
                        print("[ACTION] Next slide")
                    else:
                        action_text = "No action"

        cv2.putText(frame, f"Action: {action_text}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Gesture Slide Control - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
