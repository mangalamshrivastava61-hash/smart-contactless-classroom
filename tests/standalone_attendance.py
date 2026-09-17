"""Standalone Face Recognition Attendance test without gesture HUD."""

import time
from pathlib import Path
import cv2
import config
from src.core.face_trainer import load_labels
from src.core.smart_classroom import log_attendance_csv


def main():
    model_file = Path(config.MODEL_FILE)
    if not model_file.exists():
        print(f"[ERROR] Model file not found at {model_file}. Please train first.")
        return

    labels = load_labels(config.LABELS_FILE)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_file))

    face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INFO] Standalone Attendance running. Press 'Q' to quit.")
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
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))

            label_id, conf = recognizer.predict(face_img)
            name = labels.get(label_id, "Unknown") if conf < config.CONFIDENCE_THRESHOLD else "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
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

        cv2.putText(frame, "Standalone Attendance - Press Q to quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Standalone Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
