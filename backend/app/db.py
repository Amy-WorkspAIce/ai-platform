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
CREATE TABLE IF NOT EXISTS audio (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at REAL NOT NULL
);
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


# Columns added after the first release: (table, column, definition).
MIGRATIONS = [
    ("messages", "meta", "TEXT"),  # tool calls, tool results (JSON)
    ("conversations", "settings", "TEXT"),  # mode, workspace, web (JSON)
]


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            for table, column, definition in MIGRATIONS:
                cols = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

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
        return [_conv(r) for r in rows]

    def create_conversation(
        self, title: str, model: str | None, settings: dict | None = None
    ) -> dict:
        now = time.time()
        conv = {
            "id": new_id(),
            "title": title,
            "model": model,
            "created_at": now,
            "updated_at": now,
            "settings": json.dumps(settings or {}),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations (id, title, model, created_at, updated_at, settings) "
                "VALUES (:id, :title, :model, :created_at, :updated_at, :settings)",
                conv,
            )
        return conv | {"settings": settings or {}}

    def get_conversation(self, conv_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if row is None:
                return None
            messages = db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conv_id,),
            ).fetchall()
        return _conv(row) | {"messages": [_msg(m) for m in messages]}

    def update_conversation(self, conv_id: str, **fields) -> dict | None:
        allowed = {
            k: v for k, v in fields.items() if k in ("title", "model", "settings") and v is not None
        }
        if "settings" in allowed:
            allowed["settings"] = json.dumps(allowed["settings"])
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
        meta: dict | None = None,
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
            "meta": json.dumps(meta) if meta else None,
        }
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone():
                return None
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, reasoning, model, "
                "created_at, meta) VALUES (:id, :conversation_id, :role, :content, :reasoning, "
                ":model, :created_at, :meta)",
                msg,
            )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        return msg | {"meta": meta}

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

    # ---- videos ----

    def add_video(self, video_id: str, model: str, prompt: str, params: dict) -> dict:
        row = {
            "id": video_id,
            "model": model,
            "prompt": prompt,
            "params": json.dumps(params),
            "created_at": time.time(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO videos VALUES (:id, :model, :prompt, :params, :created_at)", row
            )
        return row | {"params": params}

    def list_videos(self, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM videos ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) | {"params": json.loads(r["params"])} for r in rows]

    def delete_video(self, video_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM videos WHERE id = ?", (video_id,)).rowcount > 0

    # ---- audio ----

    def add_audio(self, audio_id: str, model: str, text: str, params: dict) -> dict:
        row = {
            "id": audio_id,
            "model": model,
            "text": text,
            "params": json.dumps(params),
            "created_at": time.time(),
        }
        with self.connect() as db:
            db.execute("INSERT INTO audio VALUES (:id, :model, :text, :params, :created_at)", row)
        return row | {"params": params}

    def list_audio(self, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM audio ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) | {"params": json.loads(r["params"])} for r in rows]

    def delete_audio(self, audio_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM audio WHERE id = ?", (audio_id,)).rowcount > 0


def _conv(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["settings"] = json.loads(d.get("settings") or "{}")
    return d


def _msg(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["meta"] = json.loads(d["meta"]) if d.get("meta") else None
    return d
