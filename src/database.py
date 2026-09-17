"""Database manager for Smart Contactless Classroom.

Handles SQLite operations for students, teachers, and attendance records.
"""

import sqlite3
from datetime import datetime, date
import config


class Database:
    def __init__(self, db_file=None):
        self.db_file = str(db_file or config.DB_FILE)
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.make_tables()

    def make_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS students(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS teachers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    ts TEXT NOT NULL
                )
            """)

    def add_student(self, name: str):
        with self.conn:
            self.conn.execute("INSERT INTO students(name) VALUES(?)", (name,))

    def delete_student(self, sid: int):
        with self.conn:
            self.conn.execute("DELETE FROM students WHERE id=?", (sid,))

    def all_students(self):
        return self.conn.execute("SELECT id, name FROM students").fetchall()

    def add_teacher(self, name: str):
        with self.conn:
            self.conn.execute("INSERT INTO teachers(name) VALUES(?)", (name,))

    def delete_teacher(self, tid: int):
        with self.conn:
            self.conn.execute("DELETE FROM teachers WHERE id=?", (tid,))

    def all_teachers(self):
        return self.conn.execute("SELECT id, name FROM teachers").fetchall()

    def mark_attendance(self, sid: int):
        ts = datetime.now().isoformat()
        with self.conn:
            self.conn.execute("INSERT INTO attendance(student_id, ts) VALUES(?,?)", (sid, ts))

    def attendance_today(self):
        d = date.today().isoformat()
        return self.conn.execute("SELECT * FROM attendance WHERE date(ts)=?", (d,)).fetchall()

    def close(self):
        self.conn.close()
