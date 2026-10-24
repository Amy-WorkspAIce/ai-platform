# API

Interactive docs (OpenAPI) are served at `/docs` on a running instance.

All endpoints except `/health` require `Authorization: Bearer <api key>`.

## Public

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status":"ok","version":"..."}`. For container health checks. |

## OpenAI-compatible (`/v1`)

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | Models from every reachable provider, as `provider/model` IDs. |
| POST | `/v1/chat/completions` | Standard OpenAI chat request. Supports `stream: true` (SSE). The request is forwarded to the provider unchanged, except that `model` is rewritten to the upstream name. |

| POST | `/v1/images/generations` | OpenAI-style image request: `model` (`image/<tag>`), `prompt`, `n`, `size`. Extensions: `negative_prompt`, `steps`, `guidance`, `seed`. Returns `b64_json` images; they are also saved to the gallery. |
| POST | `/v1/videos/generations` | `{"model": "image/<repo>", "prompt", "negative_prompt"?, "width"?, "height"?, "frames"?, "fps"?, "steps"?, "seed"?}` → `202 {"id"}`. Runs in the background; poll `/api/videos/jobs`. |
| POST | `/v1/audio/transcriptions` | Multipart upload: `file`, `model` (`audio/whisper-...`), optional `language`, `response_format` (`json`, `text`, `verbose_json`). |
| POST | `/v1/audio/speech` | `{"model": "audio/kokoro-82m", "input": "...", "voice": "af_heart", "speed": 1.0, "response_format": "wav"}`. Returns audio (`wav`, `flac` or `ogg`). `"model": "auto"` uses the speech model chosen on the Voices page; with no `voice`, that model speaks in the voice chosen there (other models use their own default). |

`/v1/models` lists chat models plus installed image/audio models. Each entry has a
`capability`: `chat`, `image`, `transcription` or `speech`.

Planned: `/v1/embeddings`, `/v1/completions`, tool calling with connector tools.

## Management (`/api`)

| Method | Path | Description |
|---|---|---|
| GET | `/api/system` | Version and provider status. |
| GET | `/api/hardware` | Detected GPUs (VRAM), RAM, CPU count, free disk in `/data`. |
| GET | `/api/catalog` | Curated model catalog annotated with `fit`, `installed`, `needed_gb` per variant, plus `recommendations` per use case. |
| GET | `/api/catalog/models` | The whole model table in one response (one row per installable model: curated + Ollama library + top Hugging Face image/video/speech), annotated with `fit`, `installed`, `requires_token`, `downloads` (`downloads_scope`: `model`, or `family` where only per-family counts exist) and `limits` (`[{"kind": "content"|"open"|"license"|"quality", "text"}]`: what the model refuses or has no filter for, license restrictions, weak spots; curated rules matched by name, plus license and Hugging Face tags). Cached server-side; the UI fetches it once and filters/sorts locally. |
| GET | `/api/catalog/search?q=` | Search Hugging Face beyond the standard listing (GGUF chat models plus image/video/speech). Returns rows in the same shape. |
| GET | `/api/catalog/browse` | The same table filtered and sorted server-side, for API clients. Query: `category`, `q`, `sort` (`perfect`, `fit`, `size_asc`, `size_desc`, `name`, `popular`), `runnable`, `offset`, `limit`. |
| GET | `/api/settings` | Settings page data (Hugging Face status, search providers, default models, voice, version). |
| PUT/DELETE | `/api/settings/huggingface` | Save (`{"token"}`, validated with Hugging Face) or remove the Hugging Face token. |
| GET | `/api/models/installed` | Models installed in Ollama. |
| POST | `/api/models/pull` | `{"model": "qwen3:8b"}`, starts a background download (202). Accepts any Ollama tag or `hf.co/user/repo:quant`; for Hugging Face diffusers/whisper repos add `"engine": "image"` or `"audio"` (the download plan is worked out from the repo). |
| GET | `/api/models/pulls` | Progress of active and recent downloads (`status`, `completed`, `total`, `done`, `error`). |
| DELETE | `/api/models/{model}` | Deletes an installed model. |
| GET/POST | `/api/conversations` | List (most recent first) or create (`{"title", "model", "folder_id"?}`) saved chats. |
| GET/PATCH/DELETE | `/api/conversations/{id}` | Fetch with messages; PATCH `{"title"?, "folder_id"?}` renames or moves (`"folder_id": ""` removes it from its folder); delete. |
| GET | `/api/conversations/{id}/details` | The chat's Details panel: `memories` (what memory did in it, oldest first: `{id, action, text, memory_id, created_at, exists}`, `action` one of `remembered`, `updated`, `removed`, `failed`, `recalled`; `exists` is false once the fact is forgotten) and `subagents` (other models it handed work to: `{id, kind, model, task, status, result, started_at, finished_at}`, `kind` one of `ask`, `image`, `video`, `speech`; `status` one of `running`, `done`, `failed`, `stopped`). Agent chat streams the same rows live as `subagent` events. |
| GET/POST | `/api/folders` | List chat folders or create one (`{"name"}`). |
| PATCH/DELETE | `/api/folders/{id}` | `{"name"?, "collapsed"?, "position"?}`; deleting a folder keeps its chats (they become unfiled). |
| GET | `/api/memory?q=` | Remembered facts plus `enabled`, `auto`, `embedding_model`, `categories`. With `q`, ranked by relevance (by meaning when an embedding model is installed). |
| POST | `/api/memory` | Add a fact: `{"text", "category"?}`. Text that looks like a secret is rejected. |
| PATCH/DELETE | `/api/memory/{id}` | Edit (`{"text"?, "category"?}`) or forget one fact. |
| DELETE | `/api/memory?confirm=true` | Forget everything. |
| PUT | `/api/memory/settings` | `{"enabled"?, "auto"?}`: use memory in chats; learn new facts automatically. |
| POST | `/api/conversations/{id}/messages` | Append `{"role", "content", "reasoning"?, "model"?}`. |
| GET | `/api/images` | Library: images (generated or uploaded). |
| POST | `/api/images/upload` | Add your own image (multipart `file`), e.g. for the chat to edit. |
| GET | `/api/audio` | Library: speech generated by chat (or `/v1/audio/speech` with `"save": true`). |
| GET/DELETE | `/api/audio/{id}` | Fetch the WAV, or delete it. |
| PUT | `/api/settings/models` | `{"capability": "chat"|"coding"|"image"|"video"|"speech"|"transcription", "model": "<id>"|null}`: pin a model, or `null` for best installed. `chat` and `coding` are what the chat box uses on Auto. |
| GET | `/api/rules?scope=` | Rules for `global`, `folder:<id>` or `workspace:<name>`: `{scope, rules, inherited, project}`. `inherited` is the global rules, each with `excluded` for this scope; `project` is the workspace's own rule files (AGENTS.md, CLAUDE.md, .cursorrules, .cursor/rules/, .amy/rules/). |
| POST | `/api/rules` | `{"name", "text", "scope"?, "applies"?}`; `applies` is `always`, `chat` or `agent`. |
| PATCH/DELETE | `/api/rules/{id}` | Edit (`{"name"?, "text"?, "applies"?, "enabled"?}`) or delete a rule. |
| PUT | `/api/rules/{id}/excluded` | `{"scope", "excluded"}`: switch a global rule off (or back on) for one folder or workspace. |
| PUT | `/api/settings/web` | `{"enabled": bool}`: allow web access in chats. |
| PUT | `/api/settings/voice` | `{"voice": "bm_george"|null}`: the default text-to-speech voice (used when a request names none). |
| GET/DELETE | `/api/images/{id}` | Fetch the PNG, or delete it. |
| GET | `/api/audio/voices?model=audio/kokoro-82m` | A text-to-speech model's voices: `voices` (built in), `details` (display `name`, plus whatever is known of `language`, `gender` (with `estimated` when worked out from pitch), `age`, `accent`, `region`, `style`, and `group`: the set a voice belongs to), `accepts` (`clone` and/or `design`: the kinds of custom voice it can use), `needs_text` (clones need a transcript) and `custom` (the user's voices it can speak in). |
| GET | `/api/voices` | Custom voices: `id` (`custom:<id>`, used as `voice` anywhere), `name`, `kind` (`clone` or `design`), `description`, `transcript`, `seconds`. |
| POST | `/api/voices` | Add one (multipart): `name`, `kind`; for `design` a `description`; for `clone` a WAV `file` (16-bit PCM, 3 to 60 s) and ideally its `transcript`. Without a transcript, the installed speech-to-text model writes one (`transcribed: true`). |
| POST | `/api/voices/from-speech` | `{"name", "model", "voice", "text"?}`: speak a sentence with an installed model's built-in voice and keep it as a sample (for models that only copy voices, such as Qwen3-TTS Base). |
| PATCH/DELETE | `/api/voices/{id}` | Rename, change the description or transcript (`""` clears it), or delete. Deleting the default voice resets the default. |
| GET | `/api/voices/{id}/sample` | A clone's WAV sample. |
| GET | `/api/voices/library` | The voice library: `voices`, ready-made samples (`id`, `name`, `collection`, `language`, `license`, `credit`, `preview` URL, the same labels as voice `details`, `installed`: the custom voice id once installed, and `runs_on`: the installed speech models that can speak it); `models`, the installed models that can speak library voices (`how`: `built_in`, `copies` or `copies_with_text`); and, when some voice has none, `suggested`: models to install, best for this machine first. |
| GET | `/api/voices/runners` | Where each of your voices can be used: `voices` (voice id -> `[{model, voice}]`, `voice` being the id to speak with on that model: your voice, or Kyutai TTS 1.6B's built-in copy of a library voice), `models` (installed models that can use voices beyond their own, with `kinds`, `needs_text`, `built_in`) and `suggested` (per kind, `clone` or `design`, that some voice can't be used for: models to install, best for this machine first). |
| POST | `/api/voices/library` | `{"id"}`: install a library sample as a custom voice (a clone, transcribed if a speech-to-text model is installed; its `credit` is kept). |
| GET | `/api/videos/jobs` | Video jobs with live progress (`phase`, `step`, `steps`, `status`, `error`). |
| GET | `/api/videos` | Video gallery. |
| GET/DELETE | `/api/videos/{id}` | Fetch the MP4, or delete it. |
| POST | `/api/agent/chat` | The chat endpoint (tools included). Body: `model` (`auto` picks the coding or chat model per message and sends a `model` event: `{"model", "purpose"}`), `message`, `conversation_id?`, `folder_id?` (for a new chat), `mode` (`chat` or `agent`), `web?` (omit to follow the setting; `false` disables), `workspace?`, `auto_approve`. Media tools are offered for installed models; `tool_result` events may carry `attachments` (`image` id, `video` job_id, `audio` id); `notice` events report fallbacks. Streams server-sent events: `start`, `text`, `reasoning`, `tool_call`, `tool_result`, `notice`, `answer_done` (reply complete), `memory` (`{"action": "remembered"|"updated"|"removed"|"failed", "text"}`, after the answer; it finishes even if the client disconnects), `error`, `done`. Saves the conversation, including tool calls. Long chats are condensed automatically (a `notice` says so). |
| POST | `/api/agent/runs/{run_id}/approve` | `{"call_id", "approve", "always"}`: answer a pending approval. |
| GET/POST | `/api/workspaces` | List workspaces (with git branch/changes) or create one: `{"name", "git_url"?}`. |
| DELETE | `/api/workspaces/{name}` | Delete a workspace and its files. |

Installing a catalog model that needs an engine (image, audio) also installs that engine
the first time: a virtualenv in `/data/engines/<engine>` with its Python packages (several
GB for the image engine because of PyTorch). Job status shows each phase. Hugging Face
speech models (`{"model": "<repo>", "engine": "audio"}`) install the speech runtime their
library needs (`tts-chatterbox`, `tts-xtts`...; see architecture.md).

GGUF repos on Hugging Face that are **gated** (`hf.co/<repo>:<quant>`) can't be pulled by
Ollama, which has no way to send a token. Amy downloads the file with your Hugging Face token
instead, hands it to Ollama (`/api/blobs`, `/api/create`) and deletes its copy. Quantizations
split into several files aren't supported.

After a download finishes, the platform loads the model once and generates a single
token. On some GPUs, the first run compiles CUDA kernels, which can take minutes. Doing
it during install means users don't wait on their first message. The status shows
"Optimizing for your GPU (first run only)" while this happens.

Planned: provider CRUD, API key management, connectors.
