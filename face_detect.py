import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

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


        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Face Detection - Press Q To Quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
