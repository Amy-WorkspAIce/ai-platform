"""Which chat model answers: the user's pinned default for the job, else the best installed.

Two jobs: **chat** (everyday questions) and **coding** (agent mode in a workspace, and chat
messages that are plainly about code). With the chat box on **Auto**, each message goes to the
model for its job, so a coding question reaches the coding model without the user switching.
"""

import asyncio
import re

from app import catalog, hardware

PURPOSES = ("chat", "coding")
AUTO = "auto"  # the chat box's "pick for me" model

CODER = re.compile(
    r"coder|codestral|devstral|codellama|starcoder|codegemma|granite-code|codegeex|"
    r"opencoder|deepcoder|sqlcoder|\bcode\b",
    re.I,
)

# Signs a chat message is about code: fenced code, a stack trace, or programming vocabulary.
_FENCE = re.compile(r"```|^( {4}|\t)\S", re.M)
_TRACE = re.compile(
    r"Traceback \(most recent call last\)|\bat [\w.$]+\(.*:\d+\)|\w+(Error|Exception):|"
    r"\b(npm|pip|cargo|gradle|tsc) (ERR|error)",
)
_CODE_WORDS = re.compile(
    r"\b(function|regex|regexp|refactor|compile[sd]?|compiler|debug(ging)?|stack ?trace|"
    r"segfault|api endpoint|unit tests?|pull request|git (rebase|merge|commit|push)|"
    r"python|javascript|typescript|rust|golang|java|c\+\+|c#|kotlin|swift|sql|bash|"
    r"powershell|dockerfile|kubernetes|yaml|json|react|fastapi|django|node\.?js|"
    r"def |class |import |const |let |=>|\(\)\s*\{)",
    re.I,
)


def is_coder(model_id: str) -> bool:
    return bool(CODER.search(model_id.split("/", 1)[-1]))


def looks_like_code(message: str) -> bool:
    if _FENCE.search(message) or _TRACE.search(message):
        return True
    return len(set(m.group(0).lower() for m in _CODE_WORDS.finditer(message))) >= 2


async def chat_models(app) -> list[dict]:
    """Every chat model the providers offer: [{id, name, tools, coder, size_gb, fit}]."""
    providers = app.state.providers.all()
    listings = await asyncio.gather(*(p.list_details(app.state.http) for p in providers))
    s = app.state.settings
    hw = hardware.detect(s.data_dir, s.gpu_vram_gb)
    out = []
    for p, models in zip(providers, listings, strict=True):
        for m in models:
            name = m["name"]
            found = catalog.find_variant(name) if p.kind == "ollama" else None
            tools = m.get("tools")
            if tools is None and found:
                tools = bool(found[0].get("tools"))
            size = m.get("size_gb") or (found[1].get("size_gb") if found else None)
            out.append(
                {
                    "id": f"{p.name}/{name}",
                    "name": name,
                    "tools": tools,
                    "coder": is_coder(name),
                    "size_gb": size,
                    "fit": catalog.assess_fit(size, hw) if size else "unknown",
                }
            )
    return out


def rank(models: list[dict], purpose: str) -> list[dict]:
    """Best first. Runs well on this hardware beats everything; then tool use (Amy's web,
    media and workspace tools need it); then the right kind of model; then the biggest."""

    def key(m: dict):
        kind = m["coder"] if purpose == "coding" else not m["coder"]
        return (
            catalog.FIT_TIER.get(m["fit"], 3),
            m["tools"] is False,
            not kind,
            -(m["size_gb"] or 0),
        )

    return sorted(models, key=key)


async def choices(app) -> dict[str, dict]:
    """Settings-page view: per purpose, the pinned model, the best one and all of them."""
    models = await chat_models(app)
    pinned = app.state.user_settings.model_defaults()
    out = {}
    for purpose in PURPOSES:
        ranked = rank(models, purpose)
        out[purpose] = {
            "pinned": pinned.get(purpose),
            "best": ranked[0]["id"] if ranked else None,
            "installed": [{"id": m["id"], "name": m["name"], "fit": m["fit"]} for m in ranked],
        }
    return out


async def pick(app, purpose: str) -> str | None:
    """The model for a job: the pinned one while it's still available, else the best."""
    models = await chat_models(app)
    ids = {m["id"] for m in models}
    pinned = app.state.user_settings.model_defaults().get(purpose)
    if pinned in ids:
        return pinned
    ranked = rank(models, purpose)
    return ranked[0]["id"] if ranked else None
