"""Face Recognizer Model Training module.

Reads face images from the training directory, extracts labels,
and trains an OpenCV LBPH face recognizer model.
"""

from pathlib import Path
import cv2
import numpy as np
import config


def load_training_data(training_dir=None):
    """Load images and assign integer IDs per person directory."""
    training_dir = Path(training_dir or config.TRAINING_DIR)

    faces = []
    labels = []
    label_map = {}
    current_label_id = 0

    if not training_dir.exists():
        return faces, labels, label_map

    for person_folder in sorted(training_dir.iterdir()):
        if not person_folder.is_dir():
            continue

        person_name = person_folder.name
        if person_name not in label_map:
            label_map[person_name] = current_label_id
            current_label_id += 1

        label_id = label_map[person_name]

        for img_file in sorted(person_folder.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_resized = cv2.resize(img, (200, 200))
            faces.append(img_resized)
            labels.append(label_id)

    return faces, labels, label_map


def save_labels(label_map, filename=None):
    """Save label mapping (id,name) to file."""
    filename = Path(filename or config.LABELS_FILE)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for name, id_ in label_map.items():
            f.write(f"{id_},{name}\n")


def load_labels(filename=None):
    """Load label mapping (id -> name) from file."""
    filename = Path(filename or config.LABELS_FILE)
    labels = {}
    if not filename.exists():
        return labels

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def train_model(training_dir=None, model_file=None, labels_file=None):
    """Train LBPH face recognizer from training dataset."""
    training_dir = Path(training_dir or config.TRAINING_DIR)
    model_file = Path(model_file or config.MODEL_FILE)
    labels_file = Path(labels_file or config.LABELS_FILE)

    print(f"[INFO] Loading training images from: {training_dir}")
    faces, labels, label_map = load_training_data(training_dir)

    if not faces:
        raise ValueError(f"No training face images found in {training_dir}")

    print(f"[INFO] Loaded {len(faces)} images across {len(label_map)} person classes.")

    faces_np = np.array(faces, dtype="uint8")
    labels_np = np.array(labels, dtype="int32")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[INFO] Training LBPH Face Recognizer...")
    recognizer.train(faces_np, labels_np)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(model_file))
    save_labels(label_map, labels_file)

    print(f"[INFO] Model successfully saved to {model_file}")
    print(f"[INFO] Labels successfully saved to {labels_file}")
    return len(faces), len(label_map)


if __name__ == "__main__":
    train_model()
