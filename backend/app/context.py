"""Context management: keep conversations inside the model's context window.

Without this, a long chat is sent in full and the inference server silently cuts the oldest
part. Instead:

1. **Trim tool output**: old tool results (web pages, file contents, command output) are
   shortened first; they're the bulk of most long contexts and rarely needed verbatim later.
2. **Summarize**: if the history still doesn't fit, the oldest messages are folded into a
   running summary (saved on the conversation) and replaced by it.

Token counts are estimated (~3.5 characters per token), with headroom to spare.
"""

import json
import logging
import os
import re

import httpx

from app.providers import Provider

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5
RESERVE_FOR_REPLY = 2048
TRIMMED_CHARS = 600
KEEP_RECENT_TOOL_RESULTS = 3
SUMMARY_TARGET = 0.5  # after summarizing, recent history uses at most half the budget

SUMMARY_PROMPT = """Summarize the earlier part of a conversation between a user and an AI \
assistant so the assistant can continue it without the full transcript. Keep: the user's \
goals and requests, decisions made, important facts, names, numbers, file names, image/video \
ids, and anything still unresolved. Drop pleasantries and repetition. Write compact notes \
(bullets are fine), at most ~300 words. If a previous summary is given, merge it in."""


def context_size(settings) -> int:
    """The model context window: AMY_CONTEXT_TOKENS, else Ollama's configured length."""
    if settings.context_tokens:
        return settings.context_tokens
    try:
        return int(os.environ.get("OLLAMA_CONTEXT_LENGTH") or 16384)
    except ValueError:
        return 16384


def estimate_tokens(value) -> int:
    if isinstance(value, str):
        return int(len(value) / CHARS_PER_TOKEN) + 1
    return int(len(json.dumps(value, ensure_ascii=False)) / CHARS_PER_TOKEN) + 1


def message_tokens(messages: list[dict]) -> int:
    return sum(
        estimate_tokens(m.get("content") or "")
        + 4
        + (estimate_tokens(m["tool_calls"]) if m.get("tool_calls") else 0)
        for m in messages
    )


def trim_tool_results(messages: list[dict], keep_recent: int = KEEP_RECENT_TOOL_RESULTS) -> int:
    """Shorten all but the most recent tool results in place. Returns characters saved."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    saved = 0
    for i in tool_idx[:-keep_recent] if keep_recent else tool_idx:
        content = messages[i].get("content") or ""
        if len(content) > TRIMMED_CHARS + 100:
            messages[i] = {
                **messages[i],
                "content": content[:TRIMMED_CHARS]
                + f"\n[... trimmed {len(content) - TRIMMED_CHARS:,} characters of old tool output]",
            }
            saved += len(content) - TRIMMED_CHARS
    return saved


def fit(messages: list[dict], budget: int) -> None:
    """In-place, cheap: trim old tool results until the messages fit (used between agent steps)."""
    if message_tokens(messages) <= budget:
        return
    for keep in (KEEP_RECENT_TOOL_RESULTS, 1, 0):
        trim_tool_results(messages, keep)
        if message_tokens(messages) <= budget:
            return


def split_point(history: list[dict], keep_tokens: int) -> int:
    """Index where the 'recent' part starts: at a user message, keeping <= keep_tokens."""
    total = 0
    cut = len(history)
    for i in range(len(history) - 1, -1, -1):
        total += message_tokens([history[i]])
        if total > keep_tokens:
            break
        if history[i].get("role") == "user":
            cut = i
    return cut


def transcript(messages: list[dict], max_chars: int) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "tool":
            content = content[:400]
        elif role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(c["function"]["name"] for c in m["tool_calls"])
            content = f"{content}\n[called tools: {calls}]".strip()
        if content:
            lines.append(f"{role.upper()}: {content[:1500]}")
    text = "\n\n".join(lines)
    return text[-max_chars:]


async def summarize(
    client: httpx.AsyncClient,
    provider: Provider,
    model: str,
    messages: list[dict],
    previous: str | None,
    context_tokens: int,
) -> str | None:
    body = transcript(messages, max_chars=int((context_tokens - 1500) * CHARS_PER_TOKEN))
    if previous:
        body = f"PREVIOUS SUMMARY:\n{previous}\n\nCONTINUATION:\n{body}"
    try:
        resp = await client.post(
            provider.url("chat/completions"),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": body},
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 700,
            },
            headers=provider.headers(),
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        log.warning("Summarization failed: %s", exc)
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text or None


def summary_message(summary: str) -> dict:
    return {
        "role": "system",
        "content": f"Summary of the earlier part of this conversation:\n{summary}",
    }
