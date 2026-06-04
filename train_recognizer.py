import cv2
import os
import numpy as np

TRAINING_DIR = "training-data"
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.txt"

def load_training_data():
    faces = []   # list of face images (as arrays)
    labels = []  # matching numeric IDs
    label_map = {}      # name -> id (e.g. "teacher" -> 0)
    current_label_id = 0

    # loop over each person folder in training-data
    for person_name in os.listdir(TRAINING_DIR):
        person_path = os.path.join(TRAINING_DIR, person_name)
        if not os.path.isdir(person_path):
            continue  # skip files, only care about folders

        # assign an ID if this person is new
        if person_name not in label_map:
            label_map[person_name] = current_label_id
            current_label_id += 1

        label_id = label_map[person_name]

        # loop over all image files for this person
        for filename in os.listdir(person_path):
            if not (filename.lower().endswith(".jpg")
                    or filename.lower().endswith(".png")
                    or filename.lower().endswith(".jpeg")):
                continue

            img_path = os.path.join(person_path, filename)
            # read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # resize to fixed size (so model sees same size)
            img = cv2.resize(img, (200, 200))

            faces.append(img)
            labels.append(label_id)

    return faces, labels, label_map

def save_labels(label_map, filename=LABELS_FILE):
    # store as lines like: "0,teacher"
    with open(filename, "w") as f:
        for name, id_ in label_map.items():
            f.write(f"{id_},{name}\n")

def main():
    print("[INFO] Loading training images...")
    faces, labels, label_map = load_training_data()

    if not faces:
        print("[ERROR] No training data found!")
        return

    print(f"[INFO] Loaded {len(faces)} images of {len(label_map)} people.")

    faces_np = np.array(faces, dtype="uint8")
    labels_np = np.array(labels, dtype="int32")

    # Create LBPH face recognizer (part of opencv-contrib)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("[INFO] Training recognizer...")
    recognizer.train(faces_np, labels_np)

    print(f"[INFO] Saving model to {MODEL_FILE} and labels to {LABELS_FILE}")
    recognizer.save(MODEL_FILE)
    save_labels(label_map)

    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()
