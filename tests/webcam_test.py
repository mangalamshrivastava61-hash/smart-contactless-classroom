"""Webcam diagnostic test."""

import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not detected or cannot be accessed.")
        return

    print("[INFO] Webcam opened successfully. Press 'Q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            "Webcam Test - Press Q to Exit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
