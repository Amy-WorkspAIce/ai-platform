# Architecture

## Role in the suite

Amy-WorkspAIce is a set of independent apps, one repo and one container each.
The AI Platform is the **AI service** of the suite:

```
 other apps / SDKs ──(OpenAI API)──▶  AI Platform  ──▶ model engines
                                           │           (Ollama, llama.cpp, vLLM, hosted APIs)
                                           │
                                           └──(MCP tools)──▶ other apps
                                                (docs, sheets, tasks, ...)
```

- **App → AI:** apps call `/v1/chat/completions` for AI features. They can point
  at a self-hosted platform or the hosted one; the contract is the same.
- **AI → App:** apps expose MCP servers; the platform registers them as
  *connectors* and uses their tools in chats/agents (planned, see
  [connectors.md](connectors.md)).

The platform **does not implement inference**. It brokers requests to engines.

## Inside the container

```
┌──────────────────────── container (:8484) ─────────────────────────┐
│ entrypoint.sh  (PUID/PGID → drop root, supervise processes)        │
│                                                                    │
│  uvicorn ── FastAPI app ─┬─ /health          (public)              │
│                          ├─ /api/*           (management, key)     │
│                          ├─ /v1/*            (OpenAI-compatible)   │
│                          └─ /*               (React SPA, static)   │
│                                │                                   │
│                                ▼ httpx                             │
│  ollama serve (127.0.0.1:11434, only when AMY_BUNDLED_OLLAMA=true) │
│  SearXNG/granian (127.0.0.1:8888, AMY_BUNDLED_SEARXNG=true)        │
│                                                                    │
│ /data  ── secrets/, amy.db, ollama/models, images/, cache/cuda,    │
│           engines/<engine>/venv, models/<engine>/<model>           │
└────────────────────────────────────────────────────────────────────┘
```

## Code layout

```
backend/            Python 3.11+ / FastAPI
  app/main.py       app factory, router wiring, SPA serving
  app/config.py     settings (AMY_* env vars)
  app/security.py   API key generation + bearer auth
  app/providers/    model providers (OpenAI-compatible, Ollama) + registry
  app/hardware.py   GPU/RAM/disk detection (nvidia-smi, /proc/meminfo)
  app/catalog/      curated catalog (catalog.json), fit, Perfect-match ranking, browse;
                    live.py: Ollama library + Hugging Face listings, diffusers install plans
  app/ollama_admin.py  install (background pulls + GPU warm-up), list, delete
  app/jobs.py       background install jobs with pollable progress
  app/engines/      on-demand engines for images/audio: installer, worker supervisor,
                    and the worker scripts that run inside each engine's own venv
  app/db.py         SQLite: folders, conversations, messages (incl. tool calls), memories,
                    image + video galleries
  app/user_settings.py  Settings-page values (Hugging Face token in /data/secrets)
  app/agent.py      tool-calling loop, approvals, streaming events
  app/memory.py     long-term memory: fact extraction, embeddings, recall
  app/routing.py    Auto: which chat model answers (chat or coding default, else best installed)
  app/rules.py      rules (everywhere / folder / workspace), workspace rule files, ask_models
  app/context.py    context window budgeting: trim tool output, summarize old messages
  app/tools/        web (search providers, page fetch with LAN blocking), workspace
                    (confined file tools, shell commands with scrubbed env)
  app/api/          routers: system, models, conversations, agent, media, openai (/v1)
  tests/
frontend/           TypeScript / React / Vite (built into backend's static dir)
  src/views/        ChatView, LibraryView, MemoryView, ModelsView, RulesView, SettingsView,
                    VoicesView
                    (media.tsx: inline media, ChatList.tsx: sidebar folders)
scripts/            maintenance (refresh_catalog_sizes.py)
docker/             entrypoint
deploy/unraid/      Community Applications template
deploy/docker/      compose files (CPU, GPU override, external Ollama)
```

## Chat as the orchestrator

There is one interface: the chat. The chat model (any tool-capable LLM) delegates work to
specialist models through tools (`app/tools/creative.py`): image generation and editing,
video, speech, and web search. `app/delegation.py` picks the model for each kind of task
(Settings pin, else best installed by the Perfect-match ranking); `app/studio.py` holds the
operations shared with the `/v1` API. Results come back as attachments shown inline and saved
to the Library. GPU sharing: media jobs unload Ollama models first; afterwards a fully-on-GPU
pipeline is "parked" in RAM so the chat model can reload.

**Auto** (`app/routing.py`): the chat box's default model is `auto`. Agent-mode tasks, and chat
messages with code (fences, stack traces, or programming vocabulary), go to the **coding**
model; everything else to the **chat** model. Each is the Settings pin, else the best installed
chat model: runs well on this hardware first, then tool support (from Ollama's `/api/tags`
capabilities, else the catalog), then the right kind (coder or general, by name), then the
biggest. A chat that turns to code stays with the coding model for its follow-ups (the purpose
is saved on the conversation), and each message records the model that answered.

**Rules** (`app/rules.py`, `rules` and `rule_exclusions` tables): standing instructions with a
scope (`global`, `folder:<id>`, `workspace:<name>`) and a target (`always`, `chat`, `agent`).
A chat gets the global rules plus its folder's and workspace's, minus global rules that folder
or workspace switched off; they go into the system prompt widest first, and the more specific
one wins a conflict. In agent mode, rule files already in the workspace (`AGENTS.md`,
`CLAUDE.md`, `.cursorrules`, `.cursor/rules/`, `.amy/rules/`) are added too. When a rule asks
for other models' views, the `ask_models` tool is offered: one question to up to five installed
models at once, answers returned side by side. It isn't offered otherwise, because small models
reach for it on every message. Deleting a folder or workspace deletes its rules.

## Memory

Facts about the user live in the `memories` table in the same SQLite database (one short
sentence each, with a category and the chat it came from). After each reply, once the answer
has been streamed (`answer_done`), the chat model is asked for JSON `{add, update, remove}`
against the facts it already knows; results stream as `memory` events and appear in chat.
The request sees the user's last few messages too, so "remember that" works on details given
earlier, and it runs as its own task, so leaving the chat page doesn't cancel it. Background
requests (memory, summaries) send `reasoning_effort: "none"`, retried without it if the
server refuses: reasoning models otherwise think for minutes, or spend the whole token budget
thinking. When a coding model answered, a general model does the extraction (coders tend to
"remember" facts about the code). Amy's system prompt says memory exists, so it doesn't deny
it, and a failed update shows as a `memory` event with `action: "failed"`.
Anything that looks like a secret (keys, passwords, tokens, card numbers) is dropped.

Recall: with 40 facts or fewer, all of them go into the system prompt. Beyond that, the most
relevant 12 are chosen by cosine similarity of embeddings when an Ollama embedding model is
installed (vectors stored as float32 blobs next to each fact, recomputed if the model changes),
otherwise by keyword overlap. A vector database is unnecessary at personal scale: a few
thousand vectors compare in milliseconds in plain Python. `user_id` is already a column for
when multiple users arrive.

## Context management

Before each model call the chat is fitted to the context window (`AMY_CONTEXT_TOKENS`, else
`OLLAMA_CONTEXT_LENGTH`, else 16384), with room reserved for the system prompt, tool schemas
and the reply. Token counts are estimated (about 3.5 characters per token). First, old tool
results are cut down, keeping the last three intact. If the chat still doesn't fit, the oldest
messages are summarized by the chat model into a running summary saved on the conversation
(`summary`, `summary_upto`). Later turns send the summary plus the messages after it, so the
summary is only extended, never rebuilt. The full history stays in the database and the UI.

## Engines (images, audio)

Ollama only runs language models. For other modalities the platform installs an
**engine** on demand: the first time a user installs, say, an image model, the
platform creates `/data/engines/image/venv`, pip-installs PyTorch + diffusers into it, then
downloads the model to `/data/models/image/<model>`. Engines run as **worker
subprocesses** (stdlib HTTP on 127.0.0.1) that start on the first request and stop
after `AMY_ENGINE_IDLE_MINUTES` idle, freeing GPU memory. Before generating an image,
the platform asks Ollama to unload its models so the image pipeline gets the VRAM.

| Engine | Packages | Models |
|---|---|---|
| `image` | torch, diffusers, transformers, accelerate | FLUX.2 klein, SDXL, Z-Image Turbo, SD 1.5 |
| `audio` | faster-whisper (+ cuBLAS/cuDNN on GPU), kokoro-onnx | Whisper, Kokoro |

**Speech runtimes.** Most Hugging Face voice models ship with a library of their own, and
those libraries pin conflicting torch/transformers versions. A speech model's engine config
can therefore name a **runtime**: another engine with its own venv, served by
`speech_worker.py`. The model still belongs to `audio` (`audio/<repo>`, stored under
`models/audio`), so chat, read-aloud, Settings and the API treat it like Kokoro; only the
worker that loads it differs. `app/catalog/speech.py` picks the runtime from the repo's
Hugging Face `library_name` and tags, and the files to download.

| Runtime | Packages | Models |
|---|---|---|
| `tts-transformers` | torch, transformers | MMS-TTS / VITS (1,100+ languages), Bark, Sesame CSM |
| `tts-chatterbox` | chatterbox-tts | Resemble AI Chatterbox |
| `tts-f5` | f5-tts, torch/torchaudio < 2.9 | F5-TTS, E2-TTS and listed fine-tunes (plus the Vocos vocoder) |
| `tts-xtts` | coqui-tts, torch/torchaudio < 2.9 | Coqui XTTS v2 (58 built-in speakers) |
| `tts-qwen` | qwen-tts | Qwen3-TTS CustomVoice (9 speakers), VoiceDesign (described voices), Base (copies a sample) |
| `tts-omnivoice` | omnivoice | k2-fsa OmniVoice (600+ languages; described or copied voices) |
| `tts-voxcpm` | voxcpm, torch/torchaudio < 2.9 | OpenBMB VoxCPM2 (30 languages, 48 kHz; described or copied voices) |
| `tts-supertonic` | supertonic (ONNX) | Supertone Supertonic 2 and 3 (tiny, fast on CPU) |
| `tts-kitten` | kittenml | KittenTTS (ONNX) and KittenTTS 2 (47 voices; copies voices) |
| `tts-kyutai` | moshi | Kyutai TTS 1.6B (Kyutai's voice library) and 0.75B (copies voices) |
| `tts-pocket` | pocket-tts | Kyutai Pocket TTS (CPU; the gated weights also copy voices) |
| `tts-parler` | parler-tts | Parler-TTS and Indic Parler-TTS (described voices) |
| `tts-melo` | MeloTTS (GitHub), unidic-lite; Python 3.11 | MeloTTS (English accents, Chinese, Japanese, Spanish, French, Korean) |
| `tts-sopro` | sopro | Sopro (copies voices) |
| `tts-soprano` | soprano-tts | Soprano 80M |
| `tts-zonos` | Zonos (GitHub, overlaid), espeakng-loader | Zyphra Zonos transformer (copies voices) |
| `tts-neutts` | neutts, torch < 2.9, torchao 0.13, espeakng-loader | NeuTTS Air and Nano (copies voices) |
| `tts-outetts` | outetts, torch/torchaudio < 2.9 | OuteTTS 1.0 |
| `tts-indextts` | IndexTTS (GitHub); Python 3.11 | IndexTTS 2 (copies voices) |
| `tts-vibevoice` | VibeVoice (GitHub) | VibeVoice Realtime 0.5B (25 preset voices from Microsoft's repo) |
| `tts-vieneu` | vieneu, onnxruntime < 1.24 | VieNeu-TTS v3 Turbo (Vietnamese) |

Some libraries need more than a pip install, and the engine spec says so: `python` (a venv
made by `uv` with another Python, downloaded once to `engines/pythons`, for libraries with no
3.12 wheels), and `overlays` (source folders copied from a pinned GitHub commit over the
installed package, for packages that leave subpackages out). Libraries that fetch helper
models by name on first load (a BERT, a codec, voice packs) get them in the backend's
`prepare` step, right after the model downloads, into a Hugging Face cache shared by the
workers (`models/hub`, set as `HF_HUB_CACHE`); workers run offline. Models that only copy
voices offer four public-domain and CC BY **starter voices** so they can speak straight away.
Each backend's `/voices` may also return `details` (display names, languages, genders).
Backends for the newer engines live in `speech_more.py`, the shared base in `speech_base.py`.

Runtimes are big (a private Python with PyTorch, 2 to 8 GB), so deleting the last model that
uses one deletes the runtime too. Every install first checks free space (model + runtime if
it isn't installed yet + a 5 GB margin) and refuses with a clear message rather than filling
the disk.

torchaudio 2.9+ decodes audio through torchcodec, which needs system FFmpeg libraries the
image doesn't have, hence the pins. Models with a library Amy has no runtime for stay listed
as "Not supported yet", with the library named.

**Custom voices** (`app/voices.py`) are stored in the `voices` table, with clips in
`/data/voices/<id>.wav`. They are chosen by id (`custom:<id>`) like built-in voices; when
speaking, the platform sends the worker a `profile`: `{"kind": "clone", "audio", "text"}` or
`{"kind": "design", "description"}`. Each backend's `/voices` says which kinds it accepts
(`accepts`, `needs_text`), and only those voices are offered for that model; a backend given a
profile it can't use falls back to a built-in voice. The browser converts recordings and
uploads to 16-bit mono WAV before upload, so neither the platform nor the workers need
FFmpeg. OmniVoice describes voices with a fixed vocabulary (gender, age, pitch, accent,
whisper), so free descriptions are mapped onto it (`omni_tags`).

The **voice library** (`app/voice_library.py`) lists samples from Kyutai's `kyutai/tts-voices`
repo, keeping the collections that may be used commercially (CC0 and CC BY 4.0). Installing
one downloads the clip and stores it as an ordinary clone voice, with its `source` and the
`credit` its license asks for.

Pipelines are placed on the GPU in the cheapest way that fits: fully on GPU; per-component
CPU offload; **mixed**: components that fit stay on the GPU and oversized ones (e.g. a 22 GB
video text encoder) are streamed from RAM layer by layer (diffusers group offloading); or
sequential offload for very small cards. Video generation runs as a background job; the
worker exposes `GET /progress` for step-level progress.

## Model catalog

The Models page combines the curated `catalog.json` (tuned sizes and recommendations) with
live listings: the Ollama library (parsed from ollama.com, exact sizes from the Ollama
registry in the background, cached in `/data/cache/catalog`) and Hugging Face (public API,
filtered to formats Amy can run: diffusers pipelines, faster-whisper, GGUF). Families are
categorized by purpose (General, Reasoning, Code, Agent & tools, Vision, Embeddings, Image,
Video, Speech-to-text, Text-to-speech) and every model gets a fit rating. The Models page
shows one flat table (one row per installable model, e.g. "qwen3 8B"). **Perfect match**
sorting ranks the best fit tier first, then by a score: capability (memory used while running,
log scale: doubling the size is worth 1 point) plus popularity (0.12 per 10x downloads, capped
at 100M) plus a small bonus for curated picks, minus an age penalty (none for models updated in
the last 6 months, up to 1 point at 2 years). Bigger models that fit win; popularity and recency
decide between similar sizes. The same formula lives in `catalog/__init__.py` (API) and
`frontend/src/modelTable.ts` (UI). The server warms the listings at startup and refreshes them
every few hours (`AMY_CATALOG_WARMUP`); the browser fetches the table once (~40 KB gzipped)
and filters, sorts and searches locally. Hugging Face searches beyond the standard listing
are explicit ("Search Hugging Face for …").

## Model IDs

Models are exposed as `<provider>/<upstream-name>`, for example `ollama/llama3.2` or
`vllm/meta-llama/Llama-3.1-8B-Instruct`. Only the first `/` separates the two parts.
Engine models use the engine as the prefix: `image/flux2-klein-4b`, `audio/kokoro-82m`.

## Design rules

- Single port, single container, state only in `/data`.
- Zero required config; secure defaults (auth on).
- Anything another app might need goes through a documented API. Nothing is coupled through internals.
