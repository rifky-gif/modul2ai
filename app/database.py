import os
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "modul2ai.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    from app.auth import hash_password

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            uploaded_at TEXT NOT NULL
        );
    """)

    cur.execute("PRAGMA table_info(documents)")
    doc_cols = {row["name"] for row in cur.fetchall()}
    if "is_active" not in doc_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_roles (
            document_id INTEGER NOT NULL,
            role_id INTEGER NOT NULL,
            PRIMARY KEY (document_id, role_id),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS groq_usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            limit_requests INTEGER,
            remaining_requests INTEGER,
            reset_requests TEXT DEFAULT '',
            limit_tokens INTEGER,
            remaining_tokens INTEGER,
            reset_tokens TEXT DEFAULT '',
            status TEXT DEFAULT 'success',
            created_at TEXT NOT NULL
        );
    """)

    now = datetime.now(timezone.utc).isoformat()

    default_roles = [
        ("mahasiswa", "Akses informasi akademik & layanan SSC untuk mahasiswa"),
        ("dosen", "Akses informasi layanan umum serta pedoman internal khusus dosen"),
    ]
    for role_name, desc in default_roles:
        cur.execute(
            "INSERT OR IGNORE INTO roles (name, description, created_at) VALUES (?, ?, ?)",
            (role_name, desc, now),
        )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    cur.execute("SELECT id FROM admins WHERE username = ?", (admin_username,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
            (admin_username, hash_password(admin_password), now),
        )

    conn.commit()
    conn.close()
