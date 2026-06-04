import cv2
import numpy as np

MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.txt"

def load_labels(filename=LABELS_FILE):
    labels = {}
    with open(filename, "r") as f:
        for line in f:
            id_str, name = line.strip().split(",")
            labels[int(id_str)] = name
    return labels

def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("[INFO] Recognition started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label_id, confidence = recognizer.predict(face_img)

            if confidence < 80:  # threshold (tweak if needed)
                name = labels.get(label_id, "Unknown")
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
