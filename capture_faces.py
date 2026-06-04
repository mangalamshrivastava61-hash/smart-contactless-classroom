import cv2
import os

def capture_faces(name, save_dir="training-data"):
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    target_count = 20  # sample count

    print(f"[INFO] Capturing faces for: {name}")
    print("[INFO] Look at the camera.")
    print("[INFO] Press C to capture, Q to quit.")

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

            face_roi = gray[y:y+h, x:x+w]

        cv2.putText(frame, f"{name} samples: {count}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press C = capture, Q = quit",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)

        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c') and face_roi is not None:
            # save the current face
            face_img = cv2.resize(face_roi, (200, 200))
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1
            print(f"[INFO] Saved image {count}")

            if count >= target_count:
                print("[INFO] Target samples reached.")
                break

    print(f"[INFO] Done. Captured {count} images for {name}.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter person's name (folder name): ").strip()
    capture_faces(person_name)
