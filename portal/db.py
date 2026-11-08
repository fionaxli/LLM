"""SQLite storage for clients, files, chat history and practice sets."""
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "portal.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    anthropic_file_id TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,   -- exact content blocks sent to / received from the API
    display_text TEXT NOT NULL,   -- what the portal shows
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS practice_sets (
    id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    area TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    questions_json TEXT NOT NULL,  -- includes answer keys; never sent to client before submit
    answers_json TEXT,
    score INTEGER,
    total INTEGER NOT NULL,
    created_at REAL NOT NULL,
    submitted_at REAL
);
"""


def init() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def now() -> float:
    return time.time()


# ---------- clients ----------

def create_client(name: str, notes: str = "") -> dict:
    token = secrets.token_urlsafe(24)
    with conn() as c:
        cur = c.execute(
            "INSERT INTO clients (name, token, notes, created_at) VALUES (?, ?, ?, ?)",
            (name, token, notes, now()),
        )
        return get_client(cur.lastrowid, c)


def get_client(client_id: int, c=None) -> dict | None:
    if c is None:
        with conn() as c2:
            return get_client(client_id, c2)
    row = c.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    return dict(row) if row else None


def client_by_token(token: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM clients WHERE token = ?", (token,)).fetchone()
        return dict(row) if row else None


def list_clients() -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """SELECT cl.*,
                (SELECT COUNT(*) FROM files f WHERE f.client_id = cl.id AND f.deleted = 0) AS file_count,
                (SELECT COUNT(*) FROM practice_sets p WHERE p.client_id = cl.id AND p.submitted_at IS NOT NULL) AS sets_done
               FROM clients cl ORDER BY cl.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def update_client(client_id: int, name: str | None, notes: str | None) -> None:
    with conn() as c:
        if name is not None:
            c.execute("UPDATE clients SET name = ? WHERE id = ?", (name, client_id))
        if notes is not None:
            c.execute("UPDATE clients SET notes = ? WHERE id = ?", (notes, client_id))


def delete_client(client_id: int) -> list[dict]:
    """Delete a client and return their files so the caller can clean them up."""
    with conn() as c:
        files = [dict(r) for r in c.execute("SELECT * FROM files WHERE client_id = ?", (client_id,))]
        c.execute("DELETE FROM clients WHERE id = ?", (client_id,))
        return files


# ---------- files ----------

def add_file(client_id: int, filename: str, mime: str, size: int, path: str, anthropic_file_id: str) -> dict:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO files (client_id, filename, mime, size, path, anthropic_file_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (client_id, filename, mime, size, path, anthropic_file_id, now()),
        )
        return dict(c.execute("SELECT * FROM files WHERE id = ?", (cur.lastrowid,)).fetchone())


def list_files(client_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM files WHERE client_id = ? AND deleted = 0 ORDER BY created_at", (client_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_file(client_id: int, file_id: int) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM files WHERE id = ? AND client_id = ? AND deleted = 0", (file_id, client_id)
        ).fetchone()
        return dict(row) if row else None


def hide_file(client_id: int, file_id: int) -> None:
    # Soft delete: earlier chat turns still reference the uploaded copy, and
    # editing chat history would invalidate it, so the file only stops being
    # offered to *new* conversations and practice sets.
    with conn() as c:
        c.execute("UPDATE files SET deleted = 1 WHERE id = ? AND client_id = ?", (file_id, client_id))


# ---------- conversations ----------

def latest_conversation(client_id: int) -> dict:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM conversations WHERE client_id = ? ORDER BY id DESC LIMIT 1", (client_id,)
        ).fetchone()
        if row:
            return dict(row)
    return new_conversation(client_id)


def new_conversation(client_id: int) -> dict:
    with conn() as c:
        cur = c.execute("INSERT INTO conversations (client_id, created_at) VALUES (?, ?)", (client_id, now()))
        return dict(c.execute("SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)).fetchone())


def get_conversation(client_id: int, conversation_id: int) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM conversations WHERE id = ? AND client_id = ?", (conversation_id, client_id)
        ).fetchone()
        return dict(row) if row else None


def get_messages(conversation_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(conversation_id: int, role: str, content: list, display_text: str) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO messages (conversation_id, role, content_json, display_text, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, json.dumps(content, ensure_ascii=False), display_text, now()),
        )
        return cur.lastrowid


def delete_message(message_id: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM messages WHERE id = ?", (message_id,))


# ---------- practice sets ----------

def add_practice_set(client_id: int, area: str, difficulty: str, title: str, rationale: str, questions: list) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO practice_sets (client_id, area, difficulty, title, rationale, questions_json, total, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (client_id, area, difficulty, title, rationale, json.dumps(questions, ensure_ascii=False), len(questions), now()),
        )
        return cur.lastrowid


def get_practice_set(client_id: int, set_id: int) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM practice_sets WHERE id = ? AND client_id = ?", (set_id, client_id)
        ).fetchone()
        return dict(row) if row else None


def list_practice_sets(client_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM practice_sets WHERE client_id = ? ORDER BY created_at DESC", (client_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def submit_practice_set(set_id: int, answers: list, score: int) -> None:
    with conn() as c:
        c.execute(
            "UPDATE practice_sets SET answers_json = ?, score = ?, submitted_at = ? WHERE id = ?",
            (json.dumps(answers), score, now(), set_id),
        )
