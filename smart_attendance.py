import cv2
import numpy as np
import csv
import os
from datetime import datetime
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.txt")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

DEMO_MODE = True          # True = can mark many times per day
TEACHER_NAME = "teacher"  # change to your actual teacher folder name

COOLDOWN_SECONDS = 5



def load_labels(filename=LABELS_FILE):
    labels = {}
    with open(filename, "r") as f:
        for line in f:
            id_str, name = line.strip().split(",")
            labels[int(id_str)] = name
    print("[DEBUG] Loaded labels:", labels)
    return labels


def mark_attendance(name):
    """
    Writes one row: name, date, time
    DEMO_MODE: can log multiple times per day
    NORMAL: only once per day
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time"])

    already_marked = False

    if not DEMO_MODE:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    already_marked = True
                    break

    if already_marked:
        print(f"[ATTENDANCE] {name} already marked today, skipping")
        return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now_time])

    print(f"[ATTENDANCE] Marked {name} at {now_time}")



def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    print("[DEBUG] Loaded model from", MODEL_FILE)

    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("[INFO] Smart Attendance running. Press Q to quit.")
    print(f"[INFO] DEMO_MODE = {DEMO_MODE}")
    print(f"[INFO] Teacher name = {TEACHER_NAME}")
    print(f"[INFO] Cooldown per student = {COOLDOWN_SECONDS} seconds")

    last_mark_times = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label_id, confidence = recognizer.predict(face_img)

            if confidence < 80:
                name = labels.get(label_id, "Unknown")
            else:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence)})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            print("Recognized:", name, "conf:", confidence)

            if name != "Unknown" and name != TEACHER_NAME:
                now = time.time()
                last_time = last_mark_times.get(name, 0)
                if now - last_time >= COOLDOWN_SECONDS:
                    mark_attendance(name)
                    last_mark_times[name] = now
                else:
                    print(f"[DEBUG] Skipping {name}, cooldown {now - last_time:.2f}s")

        cv2.putText(frame, "Smart Attendance - Press Q to quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow("Smart Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
