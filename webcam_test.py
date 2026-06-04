import cv2

def main():
    cap = cv2.VideoCapture(0) #real time video capture uses (0)

    if not cap.isOpened():
        print("Webcam not detected!")
        return

    while True:
        success, frame = cap.read() #success reads boolean if the commands are being read or not,frame captures image as a NumPy array
        if not success:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror effect(1-flip horizontally,0-veritcally,-1-both direcn)

        cv2.imshow("Webcam Test - Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # cv2 waitkey checks if a key is pressed every millisecond, next part takes the ascii of Q and changes it to binary to sort OS differences and then exits the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
