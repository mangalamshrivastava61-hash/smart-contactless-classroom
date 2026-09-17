# Smart Contactless Classroom

A Python-based Smart Contactless Classroom System designed to automate attendance management and presentation control using computer vision, LBPH facial recognition, and MediaPipe hand gesture recognition.

[Smart Time Management Demo](https://smart-time-management-system.vercel.app/)

---

## Features

- **Automated Attendance Tracking**: Real-time face detection and recognition using OpenCV Haar Cascades and LBPH face recognizer.
- **Contactless Gesture Control**: Slide navigation and system volume control using MediaPipe hand tracking and PyAutoGUI.
- **Role Detection**: Dedicated teacher presence detection allowing teacher-only or demo-mode presentation controls.
- **Management Dashboard**: Tkinter graphical control panel to register students/teachers, view real-time logs, capture faces, and train models.
- **Multi-Format Storage**: SQLite database for structured student/teacher records and CSV logging for daily attendance exports.

---

## Technologies Used

- **Python 3.10+**
- **OpenCV (opencv-contrib-python)** (Face Detection & LBPH Recognition)
- **MediaPipe** (Hand landmark tracking & gesture detection)
- **PyAutoGUI** (Automated keyboard navigation & volume control)
- **NumPy** (Image matrix manipulations)
- **SQLite & CSV** (Database & reporting)
- **Tkinter** (Desktop User Interface)

---

## Project Structure

```text
smart-contactless-classroom/
├── config.py                 # Central configuration for paths and constants
├── main.py                   # Main application entry point (GUI / CLI)
├── requirements.txt          # Defined project dependencies
├── README.md                 # Project documentation
├── SmartClassroom.bat        # Windows launcher shortcut
├── .gitignore                # Git ignore rules
│
├── src/                      # Application source code
│   ├── __init__.py
│   ├── ui.py                 # Tkinter GUI Control Panel
│   ├── database.py           # SQLite database operations
│   └── core/                 # Core computer vision modules
│       ├── __init__.py
│       ├── smart_classroom.py # Integrated face recognition & gesture loop
│       ├── face_capture.py    # Camera face capture for training data
│       ├── face_trainer.py    # LBPH face recognizer trainer
│       └── gesture_control.py # MediaPipe hand gesture handler
│
├── data/                     # Application data
│   ├── attendance.csv        # Real-time CSV attendance log
│   ├── simple_attendance.db  # SQLite database file
│   └── training_faces/       # Dataset images organized by person
│
├── models/                   # Model storage
│   ├── face_model.yml        # Trained LBPH Face Recognizer weights
│   └── labels.txt            # Label mapping (ID -> Person Name)
│
└── tests/                    # Diagnostics & test scripts
    ├── __init__.py
    ├── webcam_test.py        # Webcam connectivity test
    ├── face_detect_test.py   # Haar cascade face detection test
    └── standalone_attendance.py # Headless attendance runner
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mangalamshrivastava61-hash/smart-contactless-classroom.git
   cd smart-contactless-classroom
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launch GUI Control Panel
```bash
python main.py
```
Or double-click `SmartClassroom.bat` on Windows.

### 2. Command-Line Options
- **Direct Camera Loop:**
  ```bash
  python main.py --classroom
  ```
- **Capture New Student Faces:**
  ```bash
  python main.py --capture "StudentName"
  ```
- **Train Recognition Model:**
  ```bash
  python main.py --train
  ```

### 3. Gesture Commands (When Active)
| Finger Count | Action |
| :--- | :--- |
| **0 fingers (fist)** | Volume Down |
| **1 finger** | Volume Up |
| **2 fingers** | Next Slide (Page Down) |
| **3 fingers** | Previous Slide (Page Up) |
| **4 fingers** | Mute / Unmute |
| **5 fingers** | Idle |

---

## Author

**Mangalam Shrivastava**  
B.Tech Computer Science Engineering  
Bennett University  
