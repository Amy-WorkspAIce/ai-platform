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
import statistics
from array import array
from collections import Counter

import httpx

from app import catalog
from app.providers import Provider
from app.tools import Tool

log = logging.getLogger(__name__)

# Facts given with each message: only ones that clearly relate to it. A fact relates when its
# meaning stands out from the rest (similarity this far above the median; measured with
# nomic-embed-text, where greetings stay under +0.08 and real matches score +0.11 to +0.21) or,
# without an embedding model, when it shares a meaningful word with the message. Anything else
# the model looks up with recall_memory when it needs it.
RECALL_LIMIT = 8
RELEVANT_MARGIN = 0.10
MIN_FOR_MARGIN = 5  # fewer facts than this make a poor median: match on words instead
LOOKUP_LIMIT = 10
KNOWN_LIMIT = 60  # extraction sees every fact up to this many
EARLIER_TURNS = 6  # the user's earlier messages in the chat shown to extraction
CATEGORIES = ("personal", "preferences", "work", "projects", "people", "setup", "goals", "general")

# Never remember these, even if mentioned.
SECRET = re.compile(
    r"(api[_ -]?key|password|passwd|secret|token|private key|ssh-|BEGIN [A-Z ]*PRIVATE|"
    r"\b(?:\d[ -]*?){13,19}\b|\bhf_[A-Za-z0-9]{10,}|\bsk-[A-Za-z0-9]{10,}|\bghp_[A-Za-z0-9]{10,})",
    re.I,
)

EXTRACT_PROMPT = """You maintain a long-term memory of durable facts about the user, so future \
conversations can be more helpful. Review the latest exchange and decide what to remember.

Remember: stable facts about the user: who they are, where they live (city and region \
only, e.g. "Lives in Redmond, WA", even when it comes up during a one-off task like a weather \
lookup), preferences (style, tools, formats, units), their work and projects, people and pets \
they mention, their hardware/software setup, goals, and explicit requests to remember something.
Do NOT remember: one-off tasks or questions, things only true right now, facts about the \
world (not the user), what the assistant said, or anything sensitive (passwords, keys, \
tokens, account or card numbers, street addresses, health or financial details) unless the \
user explicitly asks you to remember it.
Facts come only from what the USER said. Never record the assistant's claims or abilities \
(e.g. "has no persistent memory") as facts about the user. When the user asks you to remember \
something said earlier, take the facts from their earlier messages. A question about code, \
an error or a task says nothing about the user by itself: return empty lists for it.

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

INDEX_PROMPT = """You remember {count} facts about the user from earlier chats ({topics}). \
They aren't shown here. When knowing more about the user would change your answer (their \
setup, projects, preferences, people in their life) or they ask what you remember, look it up \
with recall_memory. Don't bring up what you remember when the conversation doesn't call for it, \
and never list it unprompted: a greeting gets a normal greeting."""

RECALL_HEADER = """Remembered facts that relate to this message (use them only where they help; \
if the user corrects one, go with the correction):"""

# Messages about the user as a whole ("what do you remember about me?", "based on what you
# know about me").
ABOUT_ME = re.compile(
    r"\b(know|remember|recall)\w*\s+(about\s+)?(me|myself)\b"
    r"|\babout me\b.*\?",
    re.I,
)

# Words that say nothing about a topic, so they don't make a fact match.
STOPWORDS = set(
    "the and are you your yours for with this that these those was were have has had does did "
    "doing done can could would should will shall may might must not but any all about what "
    "when where which who whom whose why how hello hey hiya there here today tonight morning "
    "evening afternoon thanks thank please okay yes yeah good great fine well just really very "
    "much more most some get got make made know like want need tell give let its it's i'm "
    "i've i'd i'll you're from into out over than then them they their our ours his her she him "
    "also too again new now".split()
)


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
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2} - STOPWORDS


def _by_words(query: str, facts: list[dict]) -> list[tuple[float, dict]]:
    q = _words(query)
    scored = [(len(q & _words(f["text"])) / (1 + len(_words(f["text"]))), f) for f in facts]
    return sorted((s for s in scored if s[0] > 0), key=lambda s: s[0], reverse=True)


async def _by_meaning(app, query: str, facts: list[dict]) -> list[tuple[float, dict]] | None:
    """Facts by similarity to ``query``, most similar first; None without an embedding model."""
    model = await embedding_model(app)
    qvec = (await embed(app, [query], model) or [None])[0] if model else None
    if not qvec:
        return None
    await ensure_embeddings(app, facts, model)
    scored = [
        (cosine(qvec, unpack(f["embedding"])), f)
        for f in facts
        if f.get("embedding") and f.get("embedding_model") == model
    ]
    return sorted(scored, key=lambda s: s[0], reverse=True)


async def recall(app, query: str, limit: int = RECALL_LIMIT) -> list[dict]:
    """The facts that clearly relate to ``query``; none for small talk."""
    db = app.state.db
    facts = db.list_memories()
    if not facts:
        return []
    if ABOUT_ME.search(query):
        # "What do you know about me?": everything, without relying on a tool call.
        chosen = facts[:50]
    elif scored := await _by_meaning(app, query, facts) if len(facts) >= MIN_FOR_MARGIN else None:
        median = statistics.median(s for s, _ in scored)
        chosen = [f for s, f in scored if s - median >= RELEVANT_MARGIN][:limit]
    else:
        chosen = [f for _, f in _by_words(query, facts)][:limit]
    db.touch_memories([f["id"] for f in chosen])
    return chosen


async def look_up(app, query: str, limit: int = LOOKUP_LIMIT) -> list[dict]:
    """recall_memory: the facts closest to ``query`` (everything, for an empty query)."""
    facts = app.state.db.list_memories()
    if not query.strip() or not facts:
        return facts[:50]
    scored = await _by_meaning(app, query, facts) or _by_words(query, facts)
    chosen = [f for _, f in scored[:limit]]
    app.state.db.touch_memories([f["id"] for f in chosen])
    return chosen


def index_prompt(facts: list[dict]) -> str | None:
    """What kinds of things memory holds, so the model knows when to look."""
    if not facts:
        return None
    topics = ", ".join(f"{c} {n}" for c, n in Counter(f["category"] for f in facts).most_common())
    return INDEX_PROMPT.format(count=len(facts), topics=topics)


def recall_tool(app, on_recall) -> Tool:
    """Look up what's remembered about the user. ``on_recall(facts)`` logs what was found."""

    async def run(args: dict) -> str:
        found = await look_up(app, str(args.get("query") or ""))
        on_recall(found)
        if not found:
            return "Nothing remembered about that."
        return "\n".join(f"- {f['text']} ({f['category']})" for f in found)

    return Tool(
        "recall_memory",
        "Look up what you remember about the user from earlier chats: their setup, projects, "
        "preferences, people and pets, anything they asked you to remember. Describe what you "
        "want to know (e.g. 'their GPU', 'pets'); leave it empty to list everything.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
        },
        run,
        summary_arg="query",
    )


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
    earlier: list[str] | None = None,
) -> list[dict]:
    """Update memory from one exchange. Returns changes: [{"action", "text", "id"}] (no id
    when the update failed)."""
    db = app.state.db
    # Every fact (or the closest when there are many), so it can update and skip duplicates.
    facts = db.list_memories()
    if len(facts) > KNOWN_LIMIT:
        scored = await _by_meaning(app, user_message, facts) or _by_words(user_message, facts)
        facts = [f for _, f in scored[:30]]
    known = facts
    known_text = "\n".join(f"{f['id']}: {f['text']}" for f in known) or "(none yet)"
    exchange = f"User: {user_message[:4000]}\n\nAssistant: {assistant_reply[:2000]}"
    if earlier:
        before = "\n\n".join(f"User: {m[:1500]}" for m in earlier)
        exchange = f"Earlier in this chat:\n{before}\n\nLatest exchange:\n{exchange}"
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
        content = await provider.quick_reply(app.state.http, payload)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        reason = str(exc) or type(exc).__name__
        log.warning("Memory update failed with %s: %s", model, reason)
        return [{"action": "failed", "text": f"{model} couldn't do it ({reason})"}]
    data = _json_object(content)
    if not data:
        log.warning("Memory update: %s didn't reply with JSON: %.300s", model, content)
        return [{"action": "failed", "text": f"{model} didn't give a usable answer"}]

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
            changes.append({"action": "removed", "text": known_ids[fid]["text"], "id": fid})
    for item in data.get("update") or []:
        if not isinstance(item, dict) or item.get("id") not in known_ids:
            continue
        text = clean_fact(item.get("text"))
        if text and text != known_ids[item["id"]]["text"]:
            emb, emb_m = await vector(text)
            db.update_memory(item["id"], text=text, embedding=emb, embedding_model=emb_m)
            changes.append({"action": "updated", "text": text, "id": item["id"]})
    # Also block re-adding facts just updated or removed (models sometimes do both).
    existing = {f["text"].lower() for f in db.list_memories()} | {f["text"].lower() for f in known}
    for item in data.get("add") or []:
        text = clean_fact(item.get("text") if isinstance(item, dict) else item)
        if not text or text.lower() in existing:
            continue
        category = item.get("category") if isinstance(item, dict) else None
        category = category if category in CATEGORIES else "general"
        emb, emb_m = await vector(text)
        fact = db.add_memory(text, category, conversation_id, emb, emb_m)
        existing.add(text.lower())
        changes.append({"action": "remembered", "text": text, "id": fact["id"]})
    return changes
