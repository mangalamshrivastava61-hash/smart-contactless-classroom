"""Smart Contactless Classroom core runtime loop.

Integrates real-time facial recognition attendance and contactless hand gesture control.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
import cv2
import config
from src.core.gesture_control import GestureController
from src.core.face_trainer import load_labels


def log_attendance_csv(name: str, attendance_file=None, demo_mode=None):
    """Log an attendance entry into the CSV file."""
    attendance_file = Path(attendance_file or config.ATTENDANCE_CSV)
    demo_mode = config.DEMO_MODE if demo_mode is None else demo_mode

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if not attendance_file.exists():
        attendance_file.parent.mkdir(parents=True, exist_ok=True)
        with open(attendance_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time"])

    if not demo_mode:
        with open(attendance_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return False

    with open(attendance_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now_time])

    print(f"[ATTENDANCE] Recorded attendance for {name} at {now_time}")
    return True


def run_classroom():
    """Main camera loop for face recognition and gesture detection."""
    model_file = Path(config.MODEL_FILE)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_file}'. Please train the model first."
        )

    labels = load_labels(config.LABELS_FILE)
    if not labels:
        print("[WARN] No labels found in labels file. Predictions will show as Unknown.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_file))

    face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
    gesture_ctrl = GestureController(cooldown=config.GESTURE_COOLDOWN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    last_mark_times = {}

    try:
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

            # Top and bottom header/footer translucent bars
            cv2.rectangle(overlay, (0, 0), (w, 70), (25, 25, 25), -1)
            cv2.rectangle(overlay, (0, h - 60), (w, h), (25, 25, 25), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # Face Detection
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
                name = (
                    labels.get(label_id, "Unknown")
                    if conf < config.CONFIDENCE_THRESHOLD
                    else "Unknown"
                )

                if name.lower() == config.TEACHER_NAME.lower():
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

                if name != "Unknown" and name.lower() != config.TEACHER_NAME.lower():
                    now = time.time()
                    last_time = last_mark_times.get(name, 0)
                    if now - last_time >= config.ATTENDANCE_COOLDOWN:
                        log_attendance_csv(name)
                        last_mark_times[name] = now

            # Teacher Status Display
            teacher_status = "Teacher: PRESENT" if teacher_present else "Teacher: NOT PRESENT"
            teacher_color = (0, 255, 0) if teacher_present else (0, 80, 255)

            cv2.putText(frame, "Smart Classroom", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, teacher_status, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, teacher_color, 2)

            mode_text = f"ATT: {'DEMO' if config.DEMO_MODE else 'REAL'} | GEST: {'DEMO' if config.GESTURE_DEMO_MODE else 'LOCKED'}"
            cv2.putText(frame, mode_text, (w - 320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Gesture Control Processing
            gestures_enabled = teacher_present or config.GESTURE_DEMO_MODE
            if gestures_enabled:
                gesture_text, gesture_color, _, _ = gesture_ctrl.process_frame(rgb, frame.shape, draw_target=frame)
            else:
                gesture_text = "Gestures: LOCKED (Teacher absent)"
                gesture_color = (50, 50, 255)

            cv2.putText(frame, f"Gesture: {gesture_text}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
            cv2.putText(frame, "0:Vol- | 1:Vol+ | 2:Next | 3:Prev | 4:Mute | Q:Quit", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Smart Classroom System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gesture_ctrl.close()


if __name__ == "__main__":
    run_classroom()
