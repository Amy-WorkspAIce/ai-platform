"""SQLite storage for conversations, messages and generated images.

Plain ``sqlite3``: one small file in the data directory, no server, no ORM. Each call opens
a short-lived connection, which is safe across FastAPI's worker threads.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    reasoning TEXT,
    model TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- conversations ----

    def list_conversations(self, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def create_conversation(self, title: str, model: str | None) -> dict:
        now = time.time()
        conv = {
            "id": new_id(),
            "title": title,
            "model": model,
            "created_at": now,
            "updated_at": now,
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (:id, :title, :model, :created_at, :updated_at)",
                conv,
            )
        return conv

    def get_conversation(self, conv_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if row is None:
                return None
            messages = db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conv_id,),
            ).fetchall()
        return dict(row) | {"messages": [dict(m) for m in messages]}

    def update_conversation(self, conv_id: str, **fields) -> dict | None:
        allowed = {k: v for k, v in fields.items() if k in ("title", "model") and v is not None}
        if allowed:
            sets = ", ".join(f"{k} = :{k}" for k in allowed)
            with self.connect() as db:
                db.execute(
                    f"UPDATE conversations SET {sets}, updated_at = :now WHERE id = :id",  # noqa: S608 - keys whitelisted
                    allowed | {"id": conv_id, "now": time.time()},
                )
        return self.get_conversation(conv_id)

    def delete_conversation(self, conv_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)).rowcount > 0

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        reasoning: str | None = None,
        model: str | None = None,
    ) -> dict | None:
        now = time.time()
        msg = {
            "id": new_id(),
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "reasoning": reasoning or None,
            "model": model,
            "created_at": now,
        }
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone():
                return None
            db.execute(
                "INSERT INTO messages VALUES "
                "(:id, :conversation_id, :role, :content, :reasoning, :model, :created_at)",
                msg,
            )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        return msg

    # ---- images ----

    def add_image(self, image_id: str, model: str, prompt: str, params: dict) -> dict:
        row = {
            "id": image_id,
            "model": model,
            "prompt": prompt,
            "params": json.dumps(params),
            "created_at": time.time(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO images VALUES (:id, :model, :prompt, :params, :created_at)", row
            )
        return row | {"params": params}

    def list_images(self, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM images ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) | {"params": json.loads(r["params"])} for r in rows]

    def delete_image(self, image_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM images WHERE id = ?", (image_id,)).rowcount > 0
