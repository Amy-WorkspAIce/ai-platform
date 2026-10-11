"""Long-term memory: durable facts about the user, learned from conversations.

- **Extraction**: after each reply, the chat model reviews the latest exchange and returns
  facts to add, update or remove (JSON). Secrets are filtered out.
- **Recall**: before each reply, the most relevant facts are added to the system prompt.
  With few facts, all of them are included; otherwise they're ranked by embedding
  similarity (if an embedding model is installed in Ollama) or keyword overlap.
- **Storage**: the ``memories`` table in SQLite, with an optional float32 embedding per fact.
  Every row has a ``user_id`` so this extends to multiple users later.
"""

import json
import logging
import math
import re
from array import array

import httpx

from app import catalog
from app.providers import Provider

log = logging.getLogger(__name__)

INCLUDE_ALL_BELOW = 40  # with this few facts, recall them all
RECALL_LIMIT = 12
CATEGORIES = ("personal", "preferences", "work", "projects", "people", "setup", "goals", "general")

# Never remember these, even if mentioned.
SECRET = re.compile(
    r"(api[_ -]?key|password|passwd|secret|token|private key|ssh-|BEGIN [A-Z ]*PRIVATE|"
    r"\b(?:\d[ -]*?){13,19}\b|\bhf_[A-Za-z0-9]{10,}|\bsk-[A-Za-z0-9]{10,}|\bghp_[A-Za-z0-9]{10,})",
    re.I,
)

EXTRACT_PROMPT = """You maintain a long-term memory of durable facts about the user, so future \
conversations can be more helpful. Review the latest exchange and decide what to remember.

Remember: stable facts about the user: who they are, preferences (style, tools, formats), \
their work and projects, people they mention, their hardware/software setup, goals, and \
explicit requests to remember something.
Do NOT remember: one-off tasks or questions, things only true right now, facts about the \
world (not the user), what the assistant said, or anything sensitive (passwords, keys, \
tokens, account or card numbers, health or financial details unless the user asks).

Write each fact as one short, self-contained statement about the user, e.g. "Prefers concise \
answers", "Runs an Unraid server with an RTX 2080 Ti", "Has a daughter named Maya".
If a known fact changed, update it instead of adding a duplicate. If the user asks to forget \
something or says a fact is wrong, remove it.

Known facts (id: text):
{known}

Reply with only JSON in this shape (use empty lists when nothing changes):
{{"add": [{{"text": "...", "category": "one of: {categories}"}}],
  "update": [{{"id": "...", "text": "..."}}],
  "remove": ["id", ...]}}"""

RECALL_HEADER = """What you know about the user from earlier conversations (use it naturally when \
relevant; don't recite it; if the user corrects something, go with the correction):"""


# ---------------------------------------------------------------- embeddings


def pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def unpack(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embedding_model_names() -> set[str]:
    """Ollama tags of the catalog's embedding models (e.g. nomic-embed-text:latest)."""
    return {
        v["tag"]
        for f in catalog.load()["families"]
        if "embedding" in f["use_cases"] and f["engine"] == "ollama"
        for v in f["variants"]
    }


async def embedding_model(app) -> str | None:
    admin = app.state.ollama_admin
    if admin is None:
        return None
    try:
        installed = {m["name"] for m in await admin.installed()}
    except httpx.HTTPError:
        return None
    known = embedding_model_names()
    for name in sorted(installed):
        base = name.split(":")[0]
        if name in known or any(k.split(":")[0] == base for k in known) or "embed" in base:
            return name
    return None


async def embed(app, texts: list[str], model: str) -> list[list[float]] | None:
    admin = app.state.ollama_admin
    try:
        resp = await admin.client.post(
            f"{admin.server_url}/api/embed", json={"model": model, "input": texts}, timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("embeddings")
    except (httpx.HTTPError, ValueError) as exc:
        log.info("Embedding failed: %s", exc)
        return None


# ---------------------------------------------------------------- recall


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


async def recall(app, query: str, limit: int = RECALL_LIMIT) -> list[dict]:
    """The facts most relevant to ``query`` (all of them when there are few)."""
    db = app.state.db
    facts = db.list_memories()
    if not facts:
        return []
    if len(facts) <= INCLUDE_ALL_BELOW:
        chosen = facts
    else:
        model = await embedding_model(app)
        scored: list[tuple[float, dict]] = []
        qvec = (await embed(app, [query], model) or [None])[0] if model else None
        if qvec:
            await ensure_embeddings(app, facts, model)
            for f in facts:
                if f.get("embedding") and f.get("embedding_model") == model:
                    scored.append((cosine(qvec, unpack(f["embedding"])), f))
        else:
            q = _words(query)
            scored = [(len(q & _words(f["text"])) / (1 + len(_words(f["text"]))), f) for f in facts]
        scored.sort(key=lambda s: s[0], reverse=True)
        chosen = [f for _, f in scored[:limit]]
    db.touch_memories([f["id"] for f in chosen])
    return chosen


async def ensure_embeddings(app, facts: list[dict], model: str) -> None:
    """Embed facts that have no embedding (or one from a different model)."""
    missing = [f for f in facts if not f.get("embedding") or f.get("embedding_model") != model]
    if not missing:
        return
    vecs = await embed(app, [f["text"] for f in missing], model)
    if not vecs:
        return
    for f, v in zip(missing, vecs, strict=False):
        f["embedding"], f["embedding_model"] = pack(v), model
        app.state.db.update_memory(f["id"], embedding=f["embedding"], embedding_model=model)


def recall_prompt(facts: list[dict]) -> str | None:
    if not facts:
        return None
    return RECALL_HEADER + "\n" + "\n".join(f"- {f['text']}" for f in facts)


async def search(app, query: str) -> list[dict]:
    """Memory page search: meaning-based if an embedding model is installed, else substring."""
    facts = app.state.db.list_memories()
    model = await embedding_model(app)
    if model and query.strip():
        qvec = (await embed(app, [query], model) or [None])[0]
        if qvec:
            await ensure_embeddings(app, facts, model)
            scored = [
                (cosine(qvec, unpack(f["embedding"])), f)
                for f in facts
                if f.get("embedding") and f.get("embedding_model") == model
            ]
            scored.sort(key=lambda s: s[0], reverse=True)
            q = query.lower()
            return [f for s, f in scored if s > 0.45 or q in f["text"].lower()]
    q = query.lower()
    return [f for f in facts if q in f["text"].lower() or q in f["category"].lower()]


# ---------------------------------------------------------------- extraction


def _json_object(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def clean_fact(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = " ".join(text.split()).strip(" -•")
    if not (3 <= len(t) <= 300) or SECRET.search(t):
        return None
    return t


async def extract(
    app,
    *,
    provider: Provider,
    model: str,
    user_message: str,
    assistant_reply: str,
    conversation_id: str,
) -> list[dict]:
    """Update memory from one exchange. Returns changes: [{"action", "text"}]."""
    db = app.state.db
    known = await recall(app, user_message, limit=30)
    known_text = "\n".join(f"{f['id']}: {f['text']}" for f in known) or "(none yet)"
    exchange = f"User: {user_message[:4000]}\n\nAssistant: {assistant_reply[:2000]}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": EXTRACT_PROMPT.format(
                    known=known_text, categories=", ".join(CATEGORIES)
                ),
            },
            {"role": "user", "content": exchange},
        ],
        "stream": False,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = await app.state.http.post(
            provider.url("chat/completions"), json=payload, headers=provider.headers(), timeout=120
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        log.info("Memory extraction skipped: %s", exc)
        return []
    data = _json_object(content)
    if not data:
        return []

    known_ids = {f["id"]: f for f in known}
    changes: list[dict] = []
    emb_model = await embedding_model(app)

    async def vector(text: str) -> tuple[bytes | None, str | None]:
        if not emb_model:
            return None, None
        v = await embed(app, [text], emb_model)
        return (pack(v[0]), emb_model) if v else (None, None)

    for fid in data.get("remove") or []:
        if isinstance(fid, str) and fid in known_ids and db.delete_memory(fid):
            changes.append({"action": "removed", "text": known_ids[fid]["text"]})
    for item in data.get("update") or []:
        if not isinstance(item, dict) or item.get("id") not in known_ids:
            continue
        text = clean_fact(item.get("text"))
        if text and text != known_ids[item["id"]]["text"]:
            emb, emb_m = await vector(text)
            db.update_memory(item["id"], text=text, embedding=emb, embedding_model=emb_m)
            changes.append({"action": "updated", "text": text})
    # Also block re-adding facts just updated or removed (models sometimes do both).
    existing = {f["text"].lower() for f in db.list_memories()} | {f["text"].lower() for f in known}
    for item in data.get("add") or []:
        text = clean_fact(item.get("text") if isinstance(item, dict) else item)
        if not text or text.lower() in existing:
            continue
        category = item.get("category") if isinstance(item, dict) else None
        category = category if category in CATEGORIES else "general"
        emb, emb_m = await vector(text)
        db.add_memory(text, category, conversation_id, emb, emb_m)
        existing.add(text.lower())
        changes.append({"action": "remembered", "text": text})
    return changes
