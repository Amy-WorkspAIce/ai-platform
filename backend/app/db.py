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
CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    collapsed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    source_conversation_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_used_at REAL,
    embedding BLOB,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS memories_by_user ON memories(user_id, updated_at);
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
CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,           -- 'clone' (a recorded sample) or 'design' (described)
    description TEXT,
    transcript TEXT,
    seconds REAL,
    created_at REAL NOT NULL
);
"""


def new_id() -> str:
    return uuid.uuid4().hex


# Columns added after the first release: (table, column, definition).
MIGRATIONS = [
    ("messages", "meta", "TEXT"),  # tool calls, tool results (JSON)
    ("conversations", "settings", "TEXT"),  # mode, workspace, web (JSON)
    ("conversations", "folder_id", "TEXT"),
    # Context management: summary of messages up to (and including) summary_upto.
    ("conversations", "summary", "TEXT"),
    ("conversations", "summary_upto", "TEXT"),
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
            k: v
            for k, v in fields.items()
            if k in ("title", "model", "settings", "folder_id", "summary", "summary_upto")
            and v is not None
        }
        if allowed.get("folder_id") == "":
            allowed["folder_id"] = None  # unfile
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

    # ---- custom voices ----

    def add_voice(
        self,
        name: str,
        kind: str,
        description: str | None = None,
        transcript: str | None = None,
        seconds: float | None = None,
        voice_id: str | None = None,
    ) -> dict:
        row = {
            "id": voice_id or new_id(),
            "name": name,
            "kind": kind,
            "description": description,
            "transcript": transcript,
            "seconds": seconds,
            "created_at": time.time(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO voices VALUES (:id, :name, :kind, :description, :transcript, "
                ":seconds, :created_at)",
                row,
            )
        return row

    def list_voices(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM voices ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def get_voice(self, voice_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone()
        return dict(row) if row else None

    def update_voice(self, voice_id: str, **fields) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in ("name", "description", "transcript")}
        if fields:
            sets = ", ".join(f"{k} = :{k}" for k in fields)
            with self.connect() as db:
                db.execute(f"UPDATE voices SET {sets} WHERE id = :id", fields | {"id": voice_id})  # noqa: S608 - fixed column names
        return self.get_voice(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM voices WHERE id = ?", (voice_id,)).rowcount > 0

    # ---- folders ----

    def list_folders(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM folders ORDER BY position, created_at").fetchall()
        return [dict(r) | {"collapsed": bool(r["collapsed"])} for r in rows]

    def create_folder(self, name: str) -> dict:
        with self.connect() as db:
            pos = db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM folders").fetchone()[0]
            row = {
                "id": new_id(),
                "name": name,
                "position": pos,
                "collapsed": 0,
                "created_at": time.time(),
            }
            db.execute(
                "INSERT INTO folders VALUES (:id, :name, :position, :collapsed, :created_at)", row
            )
        return row | {"collapsed": False}

    def update_folder(self, folder_id: str, **fields) -> dict | None:
        allowed = {
            k: v
            for k, v in fields.items()
            if k in ("name", "position", "collapsed") and v is not None
        }
        if "collapsed" in allowed:
            allowed["collapsed"] = int(bool(allowed["collapsed"]))
        with self.connect() as db:
            if allowed:
                sets = ", ".join(f"{k} = :{k}" for k in allowed)
                db.execute(f"UPDATE folders SET {sets} WHERE id = :id", allowed | {"id": folder_id})  # noqa: S608 - keys whitelisted
            row = db.execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return dict(row) | {"collapsed": bool(row["collapsed"])} if row else None

    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder; its chats are kept (unfiled)."""
        with self.connect() as db:
            db.execute(
                "UPDATE conversations SET folder_id = NULL WHERE folder_id = ?", (folder_id,)
            )
            return db.execute("DELETE FROM folders WHERE id = ?", (folder_id,)).rowcount > 0

    # ---- memories ----

    def list_memories(self, user_id: str = "default") -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def add_memory(
        self,
        text: str,
        category: str = "general",
        source: str | None = None,
        embedding: bytes | None = None,
        embedding_model: str | None = None,
        user_id: str = "default",
    ) -> dict:
        now = time.time()
        row = {
            "id": new_id(),
            "user_id": user_id,
            "text": text,
            "category": category,
            "source_conversation_id": source,
            "created_at": now,
            "updated_at": now,
            "last_used_at": None,
            "embedding": embedding,
            "embedding_model": embedding_model,
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO memories VALUES (:id, :user_id, :text, :category, "
                ":source_conversation_id, :created_at, :updated_at, :last_used_at, :embedding, "
                ":embedding_model)",
                row,
            )
        return row

    def update_memory(self, memory_id: str, **fields) -> dict | None:
        allowed = {
            k: v
            for k, v in fields.items()
            if k in ("text", "category", "embedding", "embedding_model", "last_used_at")
        }
        if "text" in allowed or "category" in allowed:
            allowed["updated_at"] = time.time()
        with self.connect() as db:
            if allowed:
                sets = ", ".join(f"{k} = :{k}" for k in allowed)
                query = f"UPDATE memories SET {sets} WHERE id = :id"  # noqa: S608 - keys whitelisted
                db.execute(query, allowed | {"id": memory_id})
            row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def touch_memories(self, ids: list[str]) -> None:
        if not ids:
            return
        with self.connect() as db:
            db.executemany(
                "UPDATE memories SET last_used_at = ? WHERE id = ?", [(time.time(), i) for i in ids]
            )

    def delete_memory(self, memory_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0

    def delete_all_memories(self, user_id: str = "default") -> int:
        with self.connect() as db:
            return db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,)).rowcount


def _conv(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["settings"] = json.loads(d.get("settings") or "{}")
    return d


def _msg(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["meta"] = json.loads(d["meta"]) if d.get("meta") else None
    return d
