"""What a model won't do: content it refuses or can't produce, and what its license forbids.

No model publishes this in a machine-readable form. Model cards and licenses describe it in
prose, and refusal behaviour is only really known from use, so these are curated rules
matched on the model's name, plus what can be read from its license and Hugging Face tags.
They apply to curated and live listings alike, so a newly listed fine-tune of a known family
inherits the family's notes.

Each limit is ``{"kind": ..., "text": ...}`` where kind is one of:
- ``content``: refuses or can't produce some kinds of content (safety tuning, filtered data)
- ``open``: guardrails removed or absent; it will produce what other models refuse
- ``license``: what the license forbids (commercial use, prohibited uses)
- ``quality``: known weak spots worth knowing before you rely on it
"""

import re

KINDS = ("content", "open", "license", "quality")

# A name that says the guardrails were removed overrides the base family's content notes.
UNCENSORED = re.compile(r"uncensored|abliterat|dolphin|\bnsfw\b|lexi\b", re.I)

CHINA = (
    "Trained under Chinese content rules: avoids, or gives the official line on, topics "
    "sensitive to the Chinese government (such as Tiananmen, Taiwan or Xinjiang)."
)
REFUSES_HARM = (
    "Refuses clearly harmful requests, such as weapons, malware or self-harm instructions."
)

# (name pattern, limits). Every matching rule applies, in order.
RULES: list[tuple[re.Pattern, list[tuple[str, str]]]] = [
    (
        re.compile(r"llama|llava", re.I),
        [
            (
                "content",
                "Safety-tuned: tends to refuse sexual content, graphic violence, and help with "
                "weapons, hacking or other illegal activity.",
            ),
            (
                "license",
                "Llama license and Acceptable Use Policy: no illegal or harmful uses; companies "
                "with over 700 million monthly users need Meta's permission.",
            ),
        ],
    ),
    (
        re.compile(r"gemma", re.I),
        [
            (
                "content",
                "Cautious safety tuning: refuses sexual content, graphic violence and dangerous "
                "instructions, and sometimes dark fiction too.",
            ),
            (
                "license",
                "Gemma Prohibited Use Policy: no sexually explicit content, harassment or other "
                "listed uses, even on your own hardware.",
            ),
        ],
    ),
    (
        re.compile(r"gpt-oss", re.I),
        [
            (
                "content",
                "Safety-tuned by OpenAI: refuses sexual content and help with weapons, malware "
                "and other harmful requests, and is hard to talk out of it.",
            ),
        ],
    ),
    (
        re.compile(r"qwen|qwq|deepseek|glm|kimi|minimax|\byi[-:\d]|internlm|baichuan", re.I),
        [("content", CHINA), ("content", REFUSES_HARM)],
    ),
    (
        re.compile(r"\bphi", re.I),
        [
            ("content", "Safety-tuned by Microsoft: refuses harmful and sexual content."),
            ("quality", "Trained mostly on English; other languages are noticeably weaker."),
        ],
    ),
    (
        re.compile(r"mistral|mixtral|ministral|magistral|devstral", re.I),
        [
            (
                "content",
                "Lightly moderated: follows your system prompt and writes mature fiction more "
                "readily than most, but still refuses clearly dangerous requests.",
            ),
        ],
    ),
    (
        re.compile(r"codestral", re.I),
        [
            (
                "license",
                "Mistral Non-Production License: for testing and research only, not "
                "commercial use.",
            )
        ],
    ),
    (
        re.compile(r"command-r|c4ai|\baya", re.I),
        [("license", "CC BY-NC: non-commercial use only, plus Cohere's acceptable use policy.")],
    ),
    (
        re.compile(r"exaone", re.I),
        [("license", "EXAONE license: research and non-commercial use only.")],
    ),
    (
        re.compile(r"granite", re.I),
        [("content", "Safety-tuned for business use. " + REFUSES_HARM)],
    ),
    (
        re.compile(r"hermes", re.I),
        [("open", "Lightly aligned: follows the system prompt with few refusals.")],
    ),
    # Image and video
    (
        re.compile(r"flux", re.I),
        [
            (
                "content",
                "Explicit material was filtered out of its training data, so it can't reliably "
                "draw nudity or sexual content.",
            )
        ],
    ),
    (
        re.compile(r"flux\.?1?[-_.]?dev|flux-dev", re.I),
        [
            (
                "license",
                "FLUX [dev] Non-Commercial License: the model can't be used commercially "
                "(the images you make can).",
            )
        ],
    ),
    (
        re.compile(r"stable-diffusion-3|\bsd3", re.I),
        [
            (
                "content",
                "Explicit material was filtered out of its training data, so it draws nudity "
                "poorly.",
            ),
            (
                "license",
                "Stability AI Community License: free under $1M annual revenue; above that you "
                "need an enterprise license.",
            ),
        ],
    ),
    (
        re.compile(r"sdxl|stable-diffusion-xl|\bsd15|stable-diffusion-v1", re.I),
        [
            (
                "open",
                "No content filter: Amy runs it without a safety checker, so it can produce "
                "nudity and other explicit images.",
            ),
            (
                "license",
                "OpenRAIL-M use restrictions: no illegal or harmful content, and no "
                "impersonating or defaming real people.",
            ),
        ],
    ),
    (
        re.compile(r"ltx", re.I),
        [
            (
                "license",
                "LTXV Open Weights License: read it before commercial use; it limits some "
                "commercial uses.",
            )
        ],
    ),
    # Speech
    (
        re.compile(r"whisper", re.I),
        [
            (
                "quality",
                "Can invent text during long silences or music, so check transcripts of quiet "
                "recordings. Less common languages are less accurate.",
            )
        ],
    ),
    (
        re.compile(r"kokoro", re.I),
        [
            (
                "quality",
                "Best in English; other languages have only a few voices. It can't clone voices.",
            )
        ],
    ),
]

NON_COMMERCIAL = re.compile(r"\bnc\b|non[- ]?commercial|research[- ]only|-nc-|-nc$", re.I)


def limits_for(family: dict) -> list[dict]:
    """The known limits of a model family (curated or live listing)."""
    # Ids carry a source prefix ("ollama:phi4"); "ollama" itself mustn't match "llama".
    key = str(family.get("id") or "").split(":", 1)[-1] + " " + str(family.get("name") or "")
    out: list[tuple[str, str]] = []
    for pattern, limits in RULES:
        if pattern.search(key):
            out += limits
    if set(family.get("use_cases") or []) == {"embedding"}:
        out = [lim for lim in out if lim[0] == "license"]  # embeddings don't write anything
    if UNCENSORED.search(key):
        out = [lim for lim in out if lim[0] != "content"]
        out.insert(
            0,
            (
                "open",
                "Guardrails removed: it answers what other models refuse, including explicit "
                "or dangerous content. You're responsible for how it's used.",
            ),
        )
    if "not-for-all-audiences" in (family.get("tags") or []) and not any(
        k == "open" for k, _ in out
    ):
        out.insert(0, ("open", "Marked as adult content on Hugging Face."))
    license = str(family.get("license") or "")
    if NON_COMMERCIAL.search(license) and not any(k == "license" for k, _ in out):
        out.append(("license", f"Non-commercial license ({license})."))
    seen: set[str] = set()
    unique = []
    for kind, text in out:
        if text not in seen:
            seen.add(text)
            unique.append({"kind": kind, "text": text})
    return unique
