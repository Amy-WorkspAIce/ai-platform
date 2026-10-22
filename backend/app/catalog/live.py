"""Live model listings: the whole Ollama library plus Hugging Face image/video/speech models.

The curated ``catalog.json`` stays the source of tuned recommendations. This module adds
everything else, normalized to the same family/variant shape so the Models page can show,
categorize, rate and install it the same way.

Sources:
- Ollama library: ollama.com/library (HTML; no official API) for names, descriptions,
  capability tags and sizes; exact download sizes from the Ollama registry (stable API),
  fetched in the background and cached.
- Hugging Face: the public models API, filtered to formats Amy's engines can run
  (diffusers pipelines, faster-whisper conversions, GGUF via Ollama).
"""

import asyncio
import json
import logging
import math
import re
import time
from fnmatch import fnmatch
from html.parser import HTMLParser
from pathlib import Path

import httpx

from app.catalog import speech

log = logging.getLogger(__name__)

OLLAMA_LIBRARY = "https://ollama.com/library"
OLLAMA_REGISTRY = "https://registry.ollama.ai/v2"
HF_API = "https://huggingface.co/api"
UA = "Mozilla/5.0 (compatible; AmyAI/1.0; +https://github.com/Amy-WorkspAIce)"

LIBRARY_TTL = 24 * 3600
HF_TTL = 6 * 3600

# Pipelines whose weights are bfloat16-native; others are fine in float16.
BF16_PIPELINES = re.compile(
    r"^(Flux|Wan|LTX|ZImage|QwenImage|HunyuanVideo|CogVideoX|StableDiffusion3|Sana|Lumina|HiDream|Chroma|Cosmos|Mochi)"
)
# Code models: by name, or descriptions that say they're *for* code (not just "good at coding").
CODE_NAME = re.compile(r"(code|coder|starcoder|devstral|codestral|granite-code)", re.I)
CODE_DESC = re.compile(
    r"\b(code generation|code completion|code model|coding model|code llm|for coding|"
    r"programming tasks|software engineering)\b",
    re.I,
)


# ---------------------------------------------------------------- helpers


def params_from_label(label: str) -> float | None:
    """'8b' -> 8.0, '137m' -> 0.137, '8x7b' -> 56, 'e4b' -> 4 (billions)."""
    m = re.fullmatch(r"(?:e)?(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?([bm])", label.lower())
    if not m:
        return None
    a = float(m.group(1)) * (float(m.group(2)) if m.group(2) else 1)
    return a if m.group(3) == "b" else a / 1000


AGE_UNITS = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30.4, "year": 365.25}


def age_days(text: str) -> float | None:
    """'yesterday' -> 1, '3 months ago' -> 91.2, 'a year ago' -> 365.25."""
    t = text.strip().lower()
    if t in ("just now", "today") or "minute" in t or "second" in t:
        return 0
    if t == "yesterday":
        return 1
    m = re.fullmatch(r"(an?|\d+)\s+(hour|day|week|month|year)s?\s+ago", t)
    if not m:
        return None
    n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
    return n * AGE_UNITS[m.group(2)]


def iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def estimate_ollama_gb(params_b: float | None) -> float | None:
    # Default Ollama tags are ~Q4_K_M: about 0.6 GB per billion parameters.
    return round(params_b * 0.6 + 0.2, 2) if params_b else None


def human_count(text: str) -> int:
    m = re.fullmatch(r"([\d.]+)\s*([KMB]?)", text.strip(), re.I)
    if not m:
        return 0
    return int(float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()])


class TTLCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, object]] = {}

    def get(self, key: str, ttl: float):
        hit = self._data.get(key)
        return hit[1] if hit and time.time() - hit[0] < ttl else None

    def put(self, key: str, value) -> None:
        self._data[key] = (time.time(), value)


# ---------------------------------------------------------------- Ollama library


class _LibraryParser(HTMLParser):
    """Parses ollama.com/library into [{name, description, capabilities, sizes, pulls}]."""

    def __init__(self) -> None:
        super().__init__()
        self.models: list[dict] = []
        self._cur: dict | None = None
        self._field: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        href = a.get("href") or ""
        if tag == "a" and re.fullmatch(r"/library/[\w.\-]+", href):
            self._cur = {
                "name": href.rsplit("/", 1)[1],
                "description": "",
                "capabilities": [],
                "sizes": [],
                "pulls": 0,
            }
            self.models.append(self._cur)
        elif self._cur is None:
            return
        elif tag == "p" and "max-w-lg" in cls:
            self._field, self._buf = "description", []
        elif tag == "span" and "rounded-md" in cls:
            if "bg-[#ddf4ff]" in cls:
                self._field, self._buf = "size", []
            elif "bg-cyan-50" in cls:
                self._field, self._buf = "cloud", []
            else:
                self._field, self._buf = "capability", []

    def handle_data(self, data):
        if self._field:
            self._buf.append(data)
        elif self._cur is not None and data.strip().endswith("Pulls"):
            pass

    def handle_endtag(self, tag):
        if not self._field or self._cur is None:
            return
        text = " ".join("".join(self._buf).split())
        if self._field == "description" and tag == "p":
            self._cur["description"] = text
        elif self._field == "size" and tag == "span":
            self._cur["sizes"].append(text)
        elif self._field == "capability" and tag == "span":
            self._cur["capabilities"].append(text.lower())
        elif self._field == "cloud" and tag == "span":
            self._cur["capabilities"].append("cloud")
        else:
            return
        self._field = None


def parse_ollama_library(html: str) -> list[dict]:
    parser = _LibraryParser()
    parser.feed(html)
    # Pull counts: "<span>39.6M</span> ... Pulls" appears after each model's tags.
    pulls = re.findall(
        r'href="/library/([\w.\-]+)".*?<span[^>]*>([\d.]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls',
        html,
        re.S,
    )
    counts = {name: human_count(n) for name, n in pulls}
    # "Updated <span>2 years ago</span>": when the model was last updated in the library.
    updated = re.findall(
        r'href="/library/([\w.\-]+)".*?Updated&nbsp;</span>\s*<span[^>]*>([^<]+)</span>', html, re.S
    )
    now = time.time()
    ages = {name: age_days(text) for name, text in updated}
    seen, out = set(), []
    for m in parser.models:
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        m["pulls"] = counts.get(m["name"], 0)
        age = ages.get(m["name"])
        m["updated"] = now - age * 86400 if age is not None else None
        out.append(m)
    return out


def ollama_family(m: dict, exact_sizes: dict[str, float]) -> dict:
    caps = set(m["capabilities"])
    local_sizes = [s for s in m["sizes"] if s != "cloud"]
    use_cases: list[str] = []
    if "embedding" in caps:
        use_cases.append("embedding")
    else:
        use_cases.append("chat")
        if "thinking" in caps:
            use_cases.append("reasoning")
        if CODE_NAME.search(m["name"]) or CODE_DESC.search(m["description"]):
            use_cases.append("code")
        if "tools" in caps:
            use_cases.append("agent")
        if "vision" in caps:
            use_cases.append("vision")
    variants = []
    for label in local_sizes or ["latest"]:
        tag = f"{m['name']}:{label}"
        params = params_from_label(label)
        exact = exact_sizes.get(tag)
        variants.append(
            {
                "tag": tag,
                "params": label.upper() if params else label,
                "size_gb": exact if exact else estimate_ollama_gb(params),
                "size_estimated": exact is None,
                # Ollama only publishes pulls per family (all sizes combined).
                "downloads": m["pulls"],
                "downloads_scope": "family",
                "updated": m.get("updated"),
            }
        )
    cloud_only = "cloud" in caps and not local_sizes
    return {
        "id": f"ollama:{m['name']}",
        "name": m["name"],
        "publisher": "Ollama library",
        "license": None,
        "description": m["description"],
        "engine": "ollama",
        "source": "ollama",
        "url": f"https://ollama.com/library/{m['name']}",
        "use_cases": use_cases,
        "tools": "tools" in caps,
        "popularity": m["pulls"],
        "unsupported": "Runs on Ollama's cloud only, not locally." if cloud_only else None,
        "variants": [] if cloud_only else variants,
    }


# ---------------------------------------------------------------- Hugging Face


HF_EXPAND = [
    "library_name",
    "gated",
    "downloads",
    "likes",
    "pipeline_tag",
    "safetensors",
    "cardData",
    "tags",
    "lastModified",
]


def hf_headers(token: str | None) -> dict:
    return {"User-Agent": UA, **({"Authorization": f"Bearer {token}"} if token else {})}


def _half_precision_gb(model: dict) -> float | None:
    total = (model.get("safetensors") or {}).get("total")
    return round(total * 2 / 1e9, 2) if total else None


def describe_hf(kind: str | None, likes: int, extra: str | None = None) -> str:
    parts = [kind] if kind else []
    if extra:
        parts.append(extra)
    parts.append(f"{likes:,} likes on Hugging Face")
    return " · ".join(parts)


def hf_family(model: dict, category: str, hf_token_set: bool) -> dict | None:
    repo = model["id"]
    tags = model.get("tags") or []
    card = model.get("cardData") or {}
    gated = model.get("gated") not in (None, False)
    diffusers_cls = next((t.split(":", 1)[1] for t in tags if t.startswith("diffusers:")), None)
    size = half = _half_precision_gb(model)
    if category == "speech" and not size:
        size = speech.KNOWN_GB.get(repo.lower())
    unsupported = None
    engine = "image"
    task = category
    if category in ("image", "video"):
        if not diffusers_cls:
            return None  # GGUF/ComfyUI repackages etc. aren't loadable by diffusers
    elif category == "transcription":
        engine = "audio"
        if "ctranslate2" not in tags and "faster-whisper" not in repo.lower():
            return None
    elif category == "speech":
        engine = "audio"
        unsupported = speech.unsupported_reason(model)
    variant = {
        "tag": repo,
        "params": f"{half / 2:.1f}B" if half else "",
        "size_gb": size,
        "size_estimated": True,
        "gated": gated,
        "requires_token": gated and not hf_token_set,
        "engine_task": task,
        "downloads": model.get("downloads", 0),
        "downloads_scope": "model",
        "updated": iso_to_epoch(model.get("lastModified")),
    }
    if size:
        # Rough until installed: components load one at a time, and the largest is usually
        # well over half of the total. Exact per-component sizes are computed at install.
        largest = size * (0.7 if engine == "image" else 1.0)
        variant.update(vram_gb=round(largest + 1.5, 1), ram_gb=round(size + 2, 1), min_vram_gb=4)
    return {
        "id": f"hf:{repo}",
        "name": repo.split("/", 1)[1],
        "publisher": repo.split("/", 1)[0],
        "license": card.get("license"),
        "tags": [t for t in tags if t == "not-for-all-audiences"],
        "description": describe_hf(
            diffusers_cls or model.get("pipeline_tag"),
            model.get("likes", 0),
            speech.voice_style(model) if category == "speech" else None,
        ),
        "engine": engine,
        "source": "huggingface",
        "url": f"https://huggingface.co/{repo}",
        "use_cases": [category],
        "tools": False,
        "popularity": model.get("downloads", 0),
        "unsupported": unsupported,
        "runtime": speech.runtime_for(model) if category == "speech" else None,
        "variants": [variant],
    }


HF_QUERIES: dict[str, list[dict]] = {
    "image": [{"pipeline_tag": "text-to-image", "library": "diffusers"}],
    "video": [
        {"pipeline_tag": "text-to-video", "library": "diffusers"},
        {"pipeline_tag": "image-to-video", "library": "diffusers"},
    ],
    "transcription": [{"search": "faster-whisper", "library": "ctranslate2"}],
    "speech": [{"pipeline_tag": "text-to-speech"}],
}


# ---------------------------------------------------------------- the live catalog


class LiveCatalog:
    def __init__(self, cache_dir: Path, client: httpx.AsyncClient, hf_token=lambda: None):
        self.cache_dir = cache_dir
        self.client = client
        self.hf_token = hf_token
        self._mem = TTLCache()
        self._sizes_file = cache_dir / "ollama-sizes.json"
        self._library_file = cache_dir / "ollama-library.json"
        self._sizes: dict[str, float] = self._load(self._sizes_file, {})
        self._size_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _load(path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _save(self, path: Path, data) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as exc:
            log.warning("Can't write catalog cache %s: %s", path, exc)

    # ---- Ollama ----

    async def ollama_models(self) -> list[dict]:
        async with self._lock:
            cached = self._load(self._library_file, None)
            if cached and time.time() - cached.get("fetched", 0) < LIBRARY_TTL:
                models = cached["models"]
            else:
                try:
                    resp = await self.client.get(
                        OLLAMA_LIBRARY, headers={"User-Agent": UA}, timeout=30
                    )
                    resp.raise_for_status()
                    models = parse_ollama_library(resp.text)
                    if len(models) < 20:
                        raise ValueError("unexpected page structure")
                    self._save(self._library_file, {"fetched": time.time(), "models": models})
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning("Ollama library unavailable (%s); using cache", exc)
                    models = (cached or {}).get("models", [])
        self._start_size_refresh(models)
        return models

    def ollama_families(self, models: list[dict]) -> list[dict]:
        return [ollama_family(m, self._sizes) for m in models]

    def _start_size_refresh(self, models: list[dict]) -> None:
        if self._size_task and not self._size_task.done():
            return
        missing = [
            (m["name"], s)
            for m in models
            for s in [x for x in m["sizes"] if x != "cloud"] or ["latest"]
            if f"{m['name']}:{s}" not in self._sizes
        ]
        if missing:
            self._size_task = asyncio.create_task(self._fetch_sizes(missing))

    async def _fetch_sizes(self, tags: list[tuple[str, str]]) -> None:
        sem = asyncio.Semaphore(8)

        async def one(name: str, tag: str) -> None:
            async with sem:
                try:
                    resp = await self.client.get(
                        f"{OLLAMA_REGISTRY}/library/{name}/manifests/{tag}",
                        headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
                        timeout=20,
                    )
                    if resp.is_success:
                        total = sum(layer["size"] for layer in resp.json().get("layers", []))
                        self._sizes[f"{name}:{tag}"] = round(total / 1e9, 2)
                except (httpx.HTTPError, ValueError, KeyError):
                    pass

        await asyncio.gather(*(one(n, t) for n, t in tags))
        self._save(self._sizes_file, self._sizes)
        log.info("Catalog: exact sizes for %d Ollama tags", len(self._sizes))

    async def keep_warm(self, interval: float = HF_TTL) -> None:
        """Fetch all standard listings at startup and refresh them periodically, so the
        Models page never waits on ollama.com or Hugging Face."""
        await asyncio.sleep(2)  # let the server finish starting
        while True:
            started = time.time()
            try:
                models = await self.ollama_models()
                for cat in HF_QUERIES:
                    await self.hf_families(cat, force=True)
                log.info(
                    "Catalog warmed: %d Ollama families (%.1fs)", len(models), time.time() - started
                )
            except Exception:  # never let the warm-up loop die
                log.exception("Catalog warm-up failed")
            await asyncio.sleep(interval)

    # ---- Hugging Face ----

    async def hf_families(
        self, category: str, query: str | None = None, limit: int = 60, force: bool = False
    ) -> list[dict]:
        token = self.hf_token()
        key = f"{category}|{query or ''}|{bool(token)}"
        hit = None if force else self._mem.get(key, HF_TTL * 2)
        if hit is not None:
            return hit
        out: dict[str, dict] = {}
        for q in HF_QUERIES.get(category, []):
            params: list[tuple[str, str]] = [
                ("sort", "downloads"),
                ("direction", "-1"),
                ("limit", str(limit)),
            ]
            params += [(k, v) for k, v in q.items()]
            if query:
                params = [p for p in params if p[0] != "search"] + [("search", query)]
            params += [("expand[]", e) for e in HF_EXPAND]
            try:
                resp = await self.client.get(
                    f"{HF_API}/models", params=params, headers=hf_headers(token), timeout=30
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Hugging Face listing failed (%s): %s", category, exc)
                continue
            for m in resp.json():
                fam = hf_family(m, category, bool(token))
                if fam and fam["id"] not in out:
                    out[fam["id"]] = fam
        result = list(out.values())
        self._mem.put(key, result)
        return result

    async def gguf_search(self, query: str, limit: int = 10) -> list[dict]:
        """Language models in GGUF format, installable through Ollama as hf.co/<repo>:<quant>."""
        token = self.hf_token()
        key = f"gguf|{query}"
        hit = self._mem.get(key, HF_TTL)
        if hit is not None:
            return hit
        try:
            resp = await self.client.get(
                f"{HF_API}/models",
                params=[
                    ("search", query),
                    ("library", "gguf"),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", str(limit)),
                    ("expand[]", "downloads"),
                    ("expand[]", "gated"),
                    ("expand[]", "cardData"),
                    ("expand[]", "lastModified"),
                ],
                headers=hf_headers(token),
                timeout=30,
            )
            resp.raise_for_status()
            models = resp.json()
        except httpx.HTTPError as exc:
            log.warning("GGUF search failed: %s", exc)
            return []

        async def detail(m: dict) -> dict | None:
            try:
                r = await self.client.get(
                    f"{HF_API}/models/{m['id']}",
                    params={"blobs": "true"},
                    headers=hf_headers(token),
                    timeout=30,
                )
                r.raise_for_status()
                siblings = r.json().get("siblings", [])
            except httpx.HTTPError:
                return None
            quants: dict[str, float] = {}
            for s in siblings:
                name = s["rfilename"]
                qm = re.search(
                    r"[-._]((?:I?Q\d[\w]*?)|F16|BF16|F32)(?:-\d+-of-\d+)?\.gguf$", name, re.I
                )
                if qm and "mmproj" not in name.lower():
                    quants[qm.group(1).upper()] = (
                        quants.get(qm.group(1).upper(), 0) + (s.get("size") or 0) / 1e9
                    )
            if not quants:
                return None
            gated = m.get("gated") not in (None, False)
            return {
                "id": f"gguf:{m['id']}",
                "name": m["id"].split("/", 1)[1],
                "publisher": m["id"].split("/", 1)[0],
                "license": (m.get("cardData") or {}).get("license"),
                "description": "GGUF from Hugging Face, runs through Ollama",
                "engine": "ollama",
                "source": "huggingface",
                "url": f"https://huggingface.co/{m['id']}",
                "use_cases": ["chat"],
                "tools": False,
                "popularity": m.get("downloads", 0),
                "unsupported": None,
                "variants": [
                    {
                        "tag": f"hf.co/{m['id']}:{q}",
                        # Gated: Amy downloads it with the user's token (Ollama can't).
                        "gated": gated,
                        "requires_token": gated and not token,
                        "params": q,
                        "size_gb": round(gb, 2),
                        "size_estimated": False,
                        "downloads": m.get("downloads", 0),
                        "downloads_scope": "family",
                        "updated": iso_to_epoch(m.get("lastModified")),
                    }
                    for q, gb in sorted(quants.items(), key=lambda kv: kv[1])
                ],
            }

        result = [f for f in await asyncio.gather(*(detail(m) for m in models)) if f]
        self._mem.put(key, result)
        return result


# ---------------------------------------------------------------- install specs


async def diffusers_spec(repo: str, client: httpx.AsyncClient, token: str | None) -> dict:
    """Work out how to download and run any diffusers-format repo.

    Picks only the files diffusers needs (fp16 weights when every component has them),
    sizes each component in half precision, and chooses defaults from the pipeline class
    and the model's own config.
    """
    headers = hf_headers(token)
    info = await client.get(
        f"{HF_API}/models/{repo}", params={"blobs": "true"}, headers=headers, timeout=30
    )
    if info.status_code in (401, 403):
        raise PermissionError("gated")
    info.raise_for_status()
    meta = info.json()
    tags = meta.get("tags") or []
    cls = next((t.split(":", 1)[1] for t in tags if t.startswith("diffusers:")), None)
    if not cls:
        raise ValueError(f"{repo} isn't in diffusers format")
    raw = f"https://huggingface.co/{repo}/resolve/main"
    index = await client.get(
        f"{raw}/model_index.json", headers=headers, timeout=30, follow_redirects=True
    )
    if index.status_code in (401, 403):
        raise PermissionError("gated")
    index.raise_for_status()
    components = [
        k
        for k, v in index.json().items()
        if not k.startswith("_") and isinstance(v, list) and v and v[0]
    ]

    files = {s["rfilename"]: (s.get("size") or 0) for s in meta.get("siblings", [])}
    weight = re.compile(r"\.(safetensors|bin)$")
    has_fp16 = {
        c: any(
            f.startswith(f"{c}/") and ".fp16." in f and f.endswith(".safetensors") for f in files
        )
        for c in components
    }
    comp_weights = {
        c: [f for f in files if f.startswith(f"{c}/") and weight.search(f)] for c in components
    }
    with_weights = [c for c in components if comp_weights[c]]
    use_fp16 = bool(with_weights) and all(has_fp16[c] for c in with_weights)

    allow = ["model_index.json"]
    sizes: dict[str, float] = {}
    for c in components:
        allow += [f"{c}/*.json", f"{c}/*.txt", f"{c}/*.model", f"{c}/*.tiktoken", f"{c}/*.jinja"]
        ws = comp_weights[c]
        if not ws:
            continue
        safetensors = [f for f in ws if f.endswith(".safetensors")]
        if use_fp16:
            chosen = [f for f in safetensors if ".fp16." in f]
        else:
            plain = [f for f in safetensors if not re.search(r"\.(fp16|bf16|fp32)\.", f)]
            chosen = plain or safetensors or ws
        allow += chosen
        sizes[c] = sum(files[f] for f in chosen) / 1e9

    # Repo weights may be fp32; estimate half-precision sizes from the repo's parameter count.
    total_bytes = sum(sizes.values()) * 1e9
    params = (meta.get("safetensors") or {}).get("total")
    ratio = 1.0
    if not use_fp16 and params and total_bytes > params * 3:  # mostly fp32 on disk
        ratio = 0.5
    half = {c: round(gb * ratio, 2) for c, gb in sizes.items()}
    dtype = "bfloat16" if BF16_PIPELINES.match(cls) else "float16"

    task = "video" if (meta.get("pipeline_tag") or "").endswith("-to-video") else "image"
    sample = await _sample_size(client, raw, headers, components)
    if task == "video":
        size = {"width": 832, "height": 480, "frames": 33, "fps": 16}
    else:
        size = {"size": sample or (1024 if BF16_PIPELINES.match(cls) or "XL" in cls else 512)}
    download_gb = round(sum(sizes.values()), 2)
    return {
        "tag": repo,
        "params": "",
        "size_gb": download_gb,
        "vram_gb": round(max(half.values(), default=0) + 2.0, 1),
        "ram_gb": round(sum(half.values()) + 2, 1),
        "engine_config": {
            "task": task,
            "pipeline": cls,
            "source": {"type": "hf", "repo": repo, "allow_patterns": sorted(set(allow))},
            "variant": "fp16" if use_fp16 else None,
            "dtype": dtype,
            # fp32 weights on disk: convert once at install so loading needs half the RAM.
            "convert": {"dtype": dtype, "keep_fp32": ["vae"] if cls.startswith("Wan") else []}
            if ratio < 1
            else None,
            "vae_fp32": cls.startswith("Wan"),
            "components_gb": half,
            "full_gb": round(sum(half.values()), 2),
            "component_gb": round(max(half.values(), default=0), 2),
            "steps": None,
            "guidance": None,
            "negative_prompt": not BF16_PIPELINES.match(cls) or cls.startswith("Wan"),
            **size,
        },
    }


async def _sample_size(
    client: httpx.AsyncClient, raw: str, headers: dict, components: list[str]
) -> int | None:
    for c in ("transformer", "unet"):
        if c not in components:
            continue
        try:
            r = await client.get(
                f"{raw}/{c}/config.json", headers=headers, timeout=20, follow_redirects=True
            )
            s = r.json().get("sample_size")
        except (httpx.HTTPError, ValueError):
            return None
        if isinstance(s, int) and s > 0:
            return s * 8 if s <= 256 else s
    return None


async def audio_spec(repo: str, client: httpx.AsyncClient, token: str | None) -> dict:
    """How to install any Hugging Face audio repo: Whisper (faster-whisper) for transcription,
    or a text-to-speech model with the runtime its library needs."""
    info = await client.get(
        f"{HF_API}/models/{repo}", params={"blobs": "true"}, headers=hf_headers(token), timeout=30
    )
    if info.status_code in (401, 403):
        raise PermissionError("gated")
    info.raise_for_status()
    meta = info.json()
    siblings = meta.get("siblings") or []
    tags = meta.get("tags") or []
    if "ctranslate2" in tags or "faster-whisper" in repo.lower():
        size = sum(s.get("size") or 0 for s in siblings) / 1e9
        return whisper_spec(repo, round(size, 2) or None)
    runtime = speech.runtime_for(meta)
    if runtime is None:
        raise ValueError(speech.unsupported_reason(meta) or f"Amy can't run {repo} yet.")
    files = [s["rfilename"] for s in siblings]
    patterns = speech.allow_patterns(runtime, repo, files)
    chosen = [
        s for s in siblings if patterns is None or any(fnmatch(s["rfilename"], p) for p in patterns)
    ]
    size = round(sum(s.get("size") or 0 for s in chosen) / 1e9, 2) or 1.0
    source: dict = {"type": "hf", "repo": repo}
    if patterns:
        source["allow_patterns"] = patterns
    return {
        "tag": repo,
        "params": "",
        "size_gb": size,
        "vram_gb": round(size * 1.3 + 1, 1),
        "ram_gb": round(size * 1.3 + 2, 1),
        "engine_config": {
            "task": "speech",
            "runtime": runtime,
            "backend": runtime.removeprefix("tts-"),
            "source": source,
            "extras": speech.extras(runtime),
        },
    }


def whisper_spec(repo: str, size_gb: float | None) -> dict:
    return {
        "tag": repo,
        "params": "",
        "size_gb": size_gb or 1.0,
        "vram_gb": round((size_gb or 1.0) + 1, 1),
        "ram_gb": round((size_gb or 1.0) + 1.5, 1),
        "engine_config": {"task": "transcription", "source": {"type": "hf", "repo": repo}},
    }


def safe_log10(n: int) -> float:
    return math.log10(n) if n > 0 else 0.0
