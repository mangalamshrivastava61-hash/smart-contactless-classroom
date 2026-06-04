import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import sqlite3, random, time, threading
from datetime import datetime, date
import csv

import capture_faces
import train_recognizer
import smart_classroom_main

DB_FILE = "simple_attendance.db"


class DB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.make_tables()

    def make_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS students(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS teachers(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                ts TEXT
            )
        """)

        self.conn.commit()

    def add_student(self, name):
        self.conn.execute("INSERT INTO students(name) VALUES(?)", (name,))
        self.conn.commit()

    def delete_student(self, sid):
        self.conn.execute("DELETE FROM students WHERE id=?", (sid,))
        self.conn.commit()

    def all_students(self):
        return self.conn.execute("SELECT id,name FROM students").fetchall()

    def add_teacher(self, name):
        self.conn.execute("INSERT INTO teachers(name) VALUES(?)", (name,))
        self.conn.commit()

    def delete_teacher(self, tid):
        self.conn.execute("DELETE FROM teachers WHERE id=?", (tid,))
        self.conn.commit()

    def all_teachers(self):
        return self.conn.execute("SELECT id,name FROM teachers").fetchall()

    def mark_attendance(self, sid):
        ts = datetime.now().isoformat()
        self.conn.execute("INSERT INTO attendance(student_id, ts) VALUES(?,?)", (sid, ts))
        self.conn.commit()

    def attendance_today(self):
        d = date.today().isoformat()
        return self.conn.execute("SELECT * FROM attendance WHERE date(ts)=?", (d,)).fetchall()


db = DB()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Smart Classroom Control Panel")
        self.geometry("900x600")

        self.running = False          # for fake canvas status
        self.current_gesture = "None"

        self.build_ui()
        self.start_thread()

    def build_ui(self):

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Dashboard", font=("Arial", 16, "bold")).pack()
        self.lbl_students = ttk.Label(left, text="Students: 0"); self.lbl_students.pack(pady=5)
        self.lbl_teachers = ttk.Label(left, text="Teachers: 0"); self.lbl_teachers.pack(pady=5)
        self.lbl_att = ttk.Label(left, text="Today's Attendance (DB): 0"); self.lbl_att.pack(pady=5)
        self.lbl_gesture = ttk.Label(left, text="Last Gesture: None"); self.lbl_gesture.pack(pady=5)

        self.canvas = tk.Canvas(left, width=400, height=250, bg="black")
        self.canvas.pack(pady=20)

        ttk.Button(left, text="Start Smart Classroom", command=self.toggle).pack(pady=5)

        ttk.Button(left, text="Capture Faces", command=self.capture_faces_gui).pack(pady=5)

        ttk.Button(left, text="Train Recognizer", command=self.train_model_gui).pack(pady=5)

        ttk.Button(left, text="Test Gesture (Fake Mark)", command=self.fake_gesture).pack(pady=5)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)

        t1 = ttk.Frame(tabs)
        tabs.add(t1, text="Students")
        self.build_students_tab(t1)

        t2 = ttk.Frame(tabs)
        tabs.add(t2, text="Teachers")
        self.build_teachers_tab(t2)

        t3 = ttk.Frame(tabs)
        tabs.add(t3, text="Attendance (DB)")
        self.build_attendance_tab(t3)

        self.refresh_all()


    def build_students_tab(self, parent):
        btns = ttk.Frame(parent); btns.pack(fill="x", pady=5)
        ttk.Button(btns, text="Add Student", command=self.add_student).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete Selected", command=self.del_student).pack(side="left")

        self.tree_students = ttk.Treeview(parent, columns=("id", "name"), show="headings")
        self.tree_students.heading("id", text="ID")
        self.tree_students.heading("name", text="Name")
        self.tree_students.pack(fill="both", expand=True)

    def add_student(self):
        name = simpledialog.askstring("Add", "Student Name:")
        if name:
            db.add_student(name)
            self.refresh_all()

    def del_student(self):
        sel = self.tree_students.selection()
        if not sel:
            return
        sid = self.tree_students.item(sel[0])["values"][0]
        db.delete_student(sid)
        self.refresh_all()


    def build_teachers_tab(self, parent):
        btns = ttk.Frame(parent); btns.pack(fill="x", pady=5)
        ttk.Button(btns, text="Add Teacher", command=self.add_teacher).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete Selected", command=self.del_teacher).pack(side="left")

        self.tree_teachers = ttk.Treeview(parent, columns=("id", "name"), show="headings")
        self.tree_teachers.heading("id", text="ID")
        self.tree_teachers.heading("name", text="Name")
        self.tree_teachers.pack(fill="both", expand=True)

    def add_teacher(self):
        name = simpledialog.askstring("Add", "Teacher Name:")
        if name:
            db.add_teacher(name)
            self.refresh_all()

    def del_teacher(self):
        sel = self.tree_teachers.selection()
        if not sel:
            return
        tid = self.tree_teachers.item(sel[0])["values"][0]
        db.delete_teacher(tid)
        self.refresh_all()


    def build_attendance_tab(self, parent):
        ttk.Button(parent, text="Export CSV (DB Attendance)", command=self.export_csv).pack(pady=5)

        self.tree_att = ttk.Treeview(parent, columns=("id", "student", "ts"), show="headings")
        for c in ("id", "student", "ts"):
            self.tree_att.heading(c, text=c.title())
        self.tree_att.pack(fill="both", expand=True)

    def export_csv(self):
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if not f:
            return
        rows = db.attendance_today()
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "student_id", "ts"])
            w.writerows(rows)
        messagebox.showinfo("Saved", "CSV Exported.")


    def capture_faces_gui(self):
        """
        Ask user for a name and call the real capture_faces.capture_faces(name)
        """
        name = simpledialog.askstring("Capture Faces", "Enter person's name (folder name):")
        if not name:
            return
        try:
            capture_faces.capture_faces(name.strip())
        except Exception as e:
            messagebox.showerror("Error", f"Error capturing faces:\n{e}")

    def train_model_gui(self):
        """
        Call train_recognizer.main() to train LBPH model.
        """
        try:
            train_recognizer.main()
            messagebox.showinfo("Train Recognizer", "Training complete.")
        except Exception as e:
            messagebox.showerror("Error", f"Error training recognizer:\n{e}")

    def toggle(self):
        if not self.running:
            self.running = True
            messagebox.showinfo(
                "Smart Classroom",
                "Starting Smart Classroom.\nPress Q in the camera window to stop."
            )
            threading.Thread(target=self._run_smart_classroom, daemon=True).start()
        else:
            messagebox.showinfo(
                "Smart Classroom",
                "Smart Classroom is already running.\nPress Q in the camera window to stop."
            )



    def fake_gesture(self):
        students = db.all_students()
        if not students:
            messagebox.showinfo("No students", "Add students first.")
            return

        sid = random.choice(students)[0]
        db.mark_attendance(sid)

        self.current_gesture = "Hand Wave (Fake)"
        self.lbl_gesture.config(text=f"Last Gesture: {self.current_gesture}")
        self.refresh_all()


    def start_thread(self):
        def loop():
            while True:
                if self.running:
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        200, 125, fill="white",
                        text=f"Smart Classroom Running...\nGesture: {self.current_gesture}"
                    )
                else:
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        200, 125, fill="white",
                        text="Camera Idle"
                    )
                time.sleep(1)

        threading.Thread(target=loop, daemon=True).start()


    def refresh_all(self):
        students = db.all_students()
        teachers = db.all_teachers()
        att = db.attendance_today()

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
    def _run_smart_classroom(self):
        try:
            smart_classroom_main.main()
        finally:
            self.running = False
            self.current_gesture = "None"
            self.lbl_gesture.config(text="Last Gesture: None")



if __name__ == "__main__":
    App().mainloop()
