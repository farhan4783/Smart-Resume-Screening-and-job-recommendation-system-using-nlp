"""
backend/database.py
--------------------
SQLite database for persisting user sessions, screening history,
and application tracking. Uses built-in sqlite3 — no server required.
"""

import os
import sqlite3
import json
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "resume_screening.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all tables. Call once at app startup."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS screening_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            candidate   TEXT,
            skills      TEXT,    -- JSON list
            match_score REAL,
            job_title   TEXT,
            mode        TEXT,    -- 'seeker' or 'recruiter'
            is_valid    INTEGER
        );

        CREATE TABLE IF NOT EXISTS job_applications (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            candidate       TEXT,
            job_title       TEXT,
            employer        TEXT,
            apply_link      TEXT,
            match_score     REAL
        );
    """)
    conn.commit()
    conn.close()


def save_screening(
    filename: str,
    candidate: str,
    skills: list[str],
    match_score: float,
    job_title: str,
    mode: str = "seeker",
    is_valid: bool = True,
):
    """Persist a resume screening result."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO screening_history
           (timestamp, filename, candidate, skills, match_score, job_title, mode, is_valid)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            filename,
            candidate,
            json.dumps(skills),
            match_score,
            job_title,
            mode,
            1 if is_valid else 0,
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 20) -> list[dict]:
    """Retrieve recent screening history."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM screening_history ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    for row in rows:
        try:
            row["skills"] = json.loads(row["skills"] or "[]")
        except Exception:
            row["skills"] = []
    return rows


def save_application(candidate: str, job_title: str, employer: str, apply_link: str, match_score: float):
    """Log a job application."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO job_applications (timestamp, candidate, job_title, employer, apply_link, match_score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), candidate, job_title, employer, apply_link, match_score),
    )
    conn.commit()
    conn.close()


def clear_history():
    """Clear all screening history (for demo resets)."""
    conn = _get_connection()
    conn.execute("DELETE FROM screening_history")
    conn.commit()
    conn.close()
