"""Central configuration module for Smart Contactless Classroom.

Provides path definitions and runtime constants.
"""

from pathlib import Path
import cv2

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRAINING_DIR = DATA_DIR / "training_faces"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Data & Model File Paths
DB_FILE = DATA_DIR / "simple_attendance.db"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"
MODEL_FILE = MODELS_DIR / "face_model.yml"
LABELS_FILE = MODELS_DIR / "labels.txt"

# Haar Cascade File
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Application Settings
DEMO_MODE = True
GESTURE_DEMO_MODE = True
TEACHER_NAME = "teacher"

ATTENDANCE_COOLDOWN = 5       # Seconds between attendance logs for the same student
GESTURE_COOLDOWN = 0.7        # Cooldown between recognized hand gestures
CONFIDENCE_THRESHOLD = 80     # Lower LBPH confidence means closer match (threshold < 80)
TARGET_SAMPLE_COUNT = 20      # Number of face snapshots during capture
