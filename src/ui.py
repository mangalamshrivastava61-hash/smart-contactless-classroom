"""Desktop Control Panel for Smart Contactless Classroom.

Provides a Tkinter-based dashboard for managing students, teachers,
attendance records, model training, and triggering the camera feed.
"""

import csv
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from pathlib import Path

import config
from src.database import Database
from src.core.face_capture import capture_faces
from src.core.face_trainer import train_model
from src.core.smart_classroom import run_classroom


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.db = Database(config.DB_FILE)

        self.title("Smart Classroom Control Panel")
        self.geometry("900x600")

        self.running = False
        self.current_gesture = "None"

        self.build_ui()
        self.start_thread()

    def build_ui(self):
        # Left Panel (Dashboard & Actions)
        left = ttk.Frame(self, padding=12)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Dashboard", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        self.lbl_students = ttk.Label(left, text="Students: 0")
        self.lbl_students.pack(pady=4, anchor="w")
        self.lbl_teachers = ttk.Label(left, text="Teachers: 0")
        self.lbl_teachers.pack(pady=4, anchor="w")
        self.lbl_att = ttk.Label(left, text="Today's Attendance (DB): 0")
        self.lbl_att.pack(pady=4, anchor="w")
        self.lbl_gesture = ttk.Label(left, text="Last Gesture: None")
        self.lbl_gesture.pack(pady=4, anchor="w")

        self.canvas = tk.Canvas(left, width=320, height=200, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(pady=15)

        ttk.Button(left, text="Start Smart Classroom", command=self.toggle_classroom).pack(fill="x", pady=4)
        ttk.Button(left, text="Capture Faces", command=self.capture_faces_gui).pack(fill="x", pady=4)
        ttk.Button(left, text="Train Recognizer", command=self.train_model_gui).pack(fill="x", pady=4)
        ttk.Button(left, text="Test Gesture (Fake Mark)", command=self.fake_gesture).pack(fill="x", pady=4)

        # Right Panel (Tabs)
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)

        t1 = ttk.Frame(tabs, padding=8)
        tabs.add(t1, text="Students")
        self.build_students_tab(t1)

        t2 = ttk.Frame(tabs, padding=8)
        tabs.add(t2, text="Teachers")
        self.build_teachers_tab(t2)

        t3 = ttk.Frame(tabs, padding=8)
        tabs.add(t3, text="Attendance (DB)")
        self.build_attendance_tab(t3)

        self.refresh_all()

    def build_students_tab(self, parent):
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=5)
        ttk.Button(btns, text="Add Student", command=self.add_student).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete Selected", command=self.del_student).pack(side="left")

        self.tree_students = ttk.Treeview(parent, columns=("id", "name"), show="headings")
        self.tree_students.heading("id", text="ID")
        self.tree_students.heading("name", text="Name")
        self.tree_students.column("id", width=60)
        self.tree_students.pack(fill="both", expand=True)

    def add_student(self):
        name = simpledialog.askstring("Add Student", "Enter student name:")
        if name and name.strip():
            self.db.add_student(name.strip())
            self.refresh_all()

    def del_student(self):
        sel = self.tree_students.selection()
        if not sel:
            return
        sid = self.tree_students.item(sel[0])["values"][0]
        self.db.delete_student(sid)
        self.refresh_all()

    def build_teachers_tab(self, parent):
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=5)
        ttk.Button(btns, text="Add Teacher", command=self.add_teacher).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete Selected", command=self.del_teacher).pack(side="left")

        self.tree_teachers = ttk.Treeview(parent, columns=("id", "name"), show="headings")
        self.tree_teachers.heading("id", text="ID")
        self.tree_teachers.heading("name", text="Name")
        self.tree_teachers.column("id", width=60)
        self.tree_teachers.pack(fill="both", expand=True)

    def add_teacher(self):
        name = simpledialog.askstring("Add Teacher", "Enter teacher name:")
        if name and name.strip():
            self.db.add_teacher(name.strip())
            self.refresh_all()

    def del_teacher(self):
        sel = self.tree_teachers.selection()
        if not sel:
            return
        tid = self.tree_teachers.item(sel[0])["values"][0]
        self.db.delete_teacher(tid)
        self.refresh_all()

    def build_attendance_tab(self, parent):
        ttk.Button(parent, text="Export CSV (DB Attendance)", command=self.export_csv).pack(anchor="w", pady=5)

        self.tree_att = ttk.Treeview(parent, columns=("id", "student_id", "ts"), show="headings")
        self.tree_att.heading("id", text="Record ID")
        self.tree_att.heading("student_id", text="Student ID")
        self.tree_att.heading("ts", text="Timestamp")
        self.tree_att.pack(fill="both", expand=True)

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        rows = self.db.attendance_today()
        with open(file_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "student_id", "timestamp"])
            writer.writerows(rows)
        messagebox.showinfo("Saved", f"Attendance successfully exported to:\n{file_path}")

    def capture_faces_gui(self):
        name = simpledialog.askstring("Capture Faces", "Enter person's name (folder name):")
        if not name or not name.strip():
            return
        try:
            count = capture_faces(name.strip())
            messagebox.showinfo("Capture Complete", f"Captured {count} samples for '{name}'.")
        except Exception as e:
            messagebox.showerror("Capture Error", f"Error capturing faces:\n{e}")

    def train_model_gui(self):
        try:
            faces_count, people_count = train_model()
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully with {faces_count} faces across {people_count} people."
            )
        except Exception as e:
            messagebox.showerror("Training Error", f"Error training recognizer:\n{e}")

    def toggle_classroom(self):
        if not self.running:
            self.running = True
            messagebox.showinfo(
                "Smart Classroom",
                "Starting Smart Classroom.\nPress 'Q' in the camera window to stop."
            )
            threading.Thread(target=self._run_smart_classroom, daemon=True).start()
        else:
            messagebox.showinfo(
                "Smart Classroom",
                "Smart Classroom is already running.\nPress 'Q' in the camera window to stop."
            )

    def _run_smart_classroom(self):
        try:
            run_classroom()
        except Exception as e:
            messagebox.showerror("Runtime Error", f"Smart Classroom error:\n{e}")
        finally:
            self.running = False
            self.current_gesture = "None"
            self.lbl_gesture.config(text="Last Gesture: None")

    def fake_gesture(self):
        students = self.db.all_students()
        if not students:
            messagebox.showinfo("No Students", "Add students first.")
            return

        sid = random.choice(students)[0]
        self.db.mark_attendance(sid)
        self.current_gesture = "Hand Wave (Simulated)"
        self.lbl_gesture.config(text=f"Last Gesture: {self.current_gesture}")
        self.refresh_all()

    def start_thread(self):
        def loop():
            while True:
                if self.running:
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        160, 100,
                        fill="#00ffcc",
                        text=f"Classroom Active\n\nGesture: {self.current_gesture}",
                        justify="center",
                        font=("Arial", 11, "bold")
                    )
                else:
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        160, 100,
                        fill="#888888",
                        text="Camera Idle",
                        justify="center",
                        font=("Arial", 11)
                    )
                time.sleep(1)

        threading.Thread(target=loop, daemon=True).start()

    def refresh_all(self):
        students = self.db.all_students()
        teachers = self.db.all_teachers()
        att = self.db.attendance_today()

        self.lbl_students.config(text=f"Students: {len(students)}")
        self.lbl_teachers.config(text=f"Teachers: {len(teachers)}")
        self.lbl_att.config(text=f"Today's Attendance (DB): {len(att)}")

        for i in self.tree_students.get_children():
            self.tree_students.delete(i)
        for s in students:
            self.tree_students.insert("", "end", values=s)

        for i in self.tree_teachers.get_children():
            self.tree_teachers.delete(i)
        for t in teachers:
            self.tree_teachers.insert("", "end", values=t)

        for i in self.tree_att.get_children():
            self.tree_att.delete(i)
        for row in att:
            self.tree_att.insert("", "end", values=row)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
