"""Face image capture module.

Captures training sample images for a given person using OpenCV Haar Cascade.
"""

from pathlib import Path
import cv2
import config


def capture_faces(name: str, save_dir=None, target_count=None):
    """Capture face sample images from webcam and save to person folder.

    Args:
        name: Name of the person.
        save_dir: Path to training data directory. Defaults to config.TRAINING_DIR.
        target_count: Total images to capture. Defaults to config.TARGET_SAMPLE_COUNT.
    """
    save_dir = Path(save_dir or config.TRAINING_DIR)
    target_count = target_count or config.TARGET_SAMPLE_COUNT

    person_dir = save_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam for face capture.")

    face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
    count = len(list(person_dir.glob("*.jpg")))

    print(f"[INFO] Capturing faces for: {name}")
    print("[INFO] Look at the camera. Press 'C' to capture, 'Q' to quit.")

    try:
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

            face_roi = None
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = gray[y:y + h, x:x + w]

            cv2.putText(
                frame,
                f"{name} samples: {count}/{target_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "Press 'C' = capture, 'Q' = quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )

            cv2.imshow("Capture Faces", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('c') and face_roi is not None:
                face_img = cv2.resize(face_roi, (200, 200))
                file_path = person_dir / f"{count}.jpg"
                cv2.imwrite(str(file_path), face_img)
                count += 1
                print(f"[INFO] Saved image {count}: {file_path.name}")

                if count >= target_count:
                    print(f"[INFO] Target samples ({target_count}) reached.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Done. Total {count} images saved for {name}.")
    return count


if __name__ == "__main__":
    person_name = input("Enter person's name: ").strip()
    if person_name:
        capture_faces(person_name)
