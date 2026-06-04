import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import csv
import os
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.txt")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

DEMO_MODE = True
GESTURE_DEMO_MODE = True

TEACHER_NAME = "teacher"

ATTENDANCE_COOLDOWN = 5
GESTURE_COOLDOWN = 0.7
CONFIDENCE_THRESHOLD = 80

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def load_labels(filename=LABELS_FILE):
    labels = {}
    with open(filename, "r") as f:
        for line in f:
            id_str, name = line.strip().split(",")
            labels[int(id_str)] = name
    return labels


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time"])

    if not DEMO_MODE:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now_time])


def count_fingers(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark

    def xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    fingers = []

    x4, y4 = xy(4)
    x2, y2 = xy(2)
    fingers.append(1 if x4 < x2 else 0)

    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        xt, yt = xy(tip_id)
        xp, yp = xy(tip_id - 2)
        fingers.append(1 if yt < yp else 0)

    return sum(fingers)


def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    last_mark_times = {}
    last_gesture_time = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, _ = frame.shape
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (w, 70), (25, 25, 25), -1)
            cv2.rectangle(overlay, (0, h - 60), (w, h), (25, 25, 25), -1)

            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(80, 80)
            )

            teacher_present = False

            for (x, y, fw, fh) in faces:
                face_img = gray[y:y + fh, x:x + fw]
                face_img = cv2.resize(face_img, (200, 200))

                label_id, conf = recognizer.predict(face_img)
                name = labels.get(label_id, "Unknown") if conf < CONFIDENCE_THRESHOLD else "Unknown"

                if name == TEACHER_NAME:
                    teacher_present = True

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({int(conf)})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                if name != "Unknown" and name != TEACHER_NAME:
                    now = time.time()
                    last_time = last_mark_times.get(name, 0)
                    if now - last_time >= ATTENDANCE_COOLDOWN:
                        mark_attendance(name)
                        last_mark_times[name] = now

            teacher_status = "Teacher: PRESENT" if teacher_present else "Teacher: NOT PRESENT"
            teacher_color = (0, 255, 0) if teacher_present else (0, 80, 255)

            title_text = "Smart Classroom"
            mode_text = f"ATT: {'DEMO' if DEMO_MODE else 'REAL'}   GEST: {'DEMO' if GESTURE_DEMO_MODE else 'LOCKED'}"

            cv2.putText(
                frame,
                title_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                teacher_status,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                teacher_color,
                2
            )

            cv2.putText(
                frame,
                mode_text,
                (w - 360, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

            gestures_enabled = (teacher_present or GESTURE_DEMO_MODE)
            gesture_text = "Gestures: DISABLED"
            gesture_color = (50, 50, 255)

            if gestures_enabled:
                gesture_text = "Gestures: SHOW HAND"
                gesture_color = (0, 255, 255)

                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                        fingers_up = count_fingers(handLms, frame.shape)
                        now = time.time()

                        if now - last_gesture_time >= GESTURE_COOLDOWN:
                            if fingers_up == 0:
                                pyautogui.press("volumedown")
                                gesture_text = "Volume Down"
                                gesture_color = (0, 255, 255)
                                last_gesture_time = now
                            elif fingers_up == 1:
                                pyautogui.press("volumeup")
                                gesture_text = "Volume Up"
                                gesture_color = (0, 255, 255)
                                last_gesture_time = now
                            elif fingers_up == 2:
                                pyautogui.press("pagedown")
                                gesture_text = "Next Slide"
                                gesture_color = (0, 255, 255)
                                last_gesture_time = now
                            elif fingers_up == 3:
                                pyautogui.press("pageup")
                                gesture_text = "Previous Slide"
                                gesture_color = (0, 255, 255)
                                last_gesture_time = now
                            elif fingers_up == 4:
                                pyautogui.press("volumemute")
                                gesture_text = "Mute/Unmute"
                                gesture_color = (0, 255, 255)
                                last_gesture_time = now
                            else:
                                gesture_text = "Idle"
                                gesture_color = (0, 255, 0)
                        else:
                            gesture_text = "Cooldown"
                            gesture_color = (0, 255, 255)

            cv2.putText(
                frame,
                gesture_text,
                (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                gesture_color,
                2
            )

            cv2.putText(
                frame,
                "0:Vol-  1:Vol+  2:Next  3:Prev  4:Mute  5:Idle   |   Q = Quit",
                (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )

            cv2.imshow("Smart Classroom System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
