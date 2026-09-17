"""Main entry point for Smart Contactless Classroom.

Supports launching the graphical desktop control panel or direct CLI commands.
Usage:
    python main.py                     # Launch GUI Control Panel (Default)
    python main.py --classroom         # Launch Smart Classroom Camera directly
    python main.py --train             # Train face recognizer model
    python main.py --capture <NAME>    # Capture face samples for person
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Smart Contactless Classroom System"
    )
    parser.add_argument(
        "--classroom",
        action="store_true",
        help="Start the smart classroom camera recognition and gesture loop directly"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the LBPH face recognizer with images in data/training_faces"
    )
    parser.add_argument(
        "--capture",
        type=str,
        metavar="NAME",
        help="Capture face samples for the specified person"
    )

    args = parser.parse_args()

    if args.classroom:
        from src.core.smart_classroom import run_classroom
        run_classroom()
    elif args.train:
        from src.core.face_trainer import train_model
        train_model()
    elif args.capture:
        from src.core.face_capture import capture_faces
        capture_faces(args.capture)
    else:
        # Default: Launch GUI Dashboard
        from src.ui import main as launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
