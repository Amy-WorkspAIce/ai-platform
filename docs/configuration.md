# Configuration

All settings are environment variables. None are required.

| Variable | Default | Description |
|---|---|---|
| `AMY_API_KEY` | *(auto)* | API key for `/v1` and `/api`. If unset, one is generated on first start, saved to `/data/secrets/api_key`, and printed to the logs. |
| `AMY_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server. The default is the bundled one. Set to empty to disable the Ollama provider. |
| `AMY_BUNDLED_OLLAMA` | `true` (full image) / `false` (slim) | Start the Ollama engine inside the container. |
| `AMY_PROVIDERS` | `[]` | Extra OpenAI-compatible providers, as a JSON list of `{"name", "base_url", "api_key"?}`. `base_url` is the API root (usually ends in `/v1`). |
| `AMY_CORS_ORIGINS` | *(none)* | Comma-separated browser origins allowed to call the API (for other apps' frontends). |
| `AMY_GPU_VRAM_GB` | *(detected)* | Override detected GPU memory for model recommendations. Useful when Ollama runs on another machine (slim image) or the GPU isn't NVIDIA. |
| `AMY_ENGINE_IDLE_MINUTES` | `10` | Stop idle image/audio engine workers after this long, returning their RAM and VRAM to language models. |
| `AMY_TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu128` | PyTorch wheel index for the image engine on NVIDIA GPUs. cu128 needs driver 570+; use `.../whl/cu126` for older drivers. Machines without an NVIDIA GPU use the CPU wheels. |
| `AMY_SEARCH_PROVIDER` | `auto` | Web search: `auto`, `searxng`, `brave` or `duckduckgo`. `auto` uses the first one configured, falling back to DuckDuckGo. |
| `AMY_BUNDLED_SEARXNG` | `true` (full image) | Run SearXNG inside the container for web search. |
| `AMY_SEARXNG_URL` | *(bundled)* | Use your own SearXNG instead, e.g. `http://192.168.1.10:8080` (JSON output must be enabled). Blank means the bundled one. |
| `AMY_BRAVE_API_KEY` | *(none)* | Brave Search API key. |
| `AMY_WEB_ALLOW_PRIVATE` | `false` | Let `fetch_url` reach private/LAN addresses. |
| `AMY_HF_TOKEN` | *(none)* | Hugging Face token for gated models. Normally set on the **Settings** page; this variable overrides it and makes it read-only there. |
| `AMY_CATALOG_WARMUP` | `true` | Fetch model listings (Ollama library, Hugging Face) at startup and refresh them periodically. |
| `AMY_GITHUB_TOKEN` | *(none)* | Token for agent git operations on github.com (clone private repos, push). Use a fine-grained token. |
| `AMY_AGENT_MAX_STEPS` | `40` | Maximum model turns per agent request. |
| `AMY_AGENT_COMMAND_TIMEOUT` | `600` | Maximum seconds per agent command. |
| `AMY_CONTEXT_TOKENS` | `0` | Context window the chat is fitted to, in tokens. `0` uses `OLLAMA_CONTEXT_LENGTH` (or 16384). Set it when using an external engine with a different window. |
| `OLLAMA_CONTEXT_LENGTH` | `16384` | Context size for chat models (full image). Agent mode needs 16k+. Flash attention and a q8_0 KV cache are enabled to keep memory use down. |
| `AMY_UPSTREAM_TIMEOUT` | `600` | Seconds to wait on a model engine. |
| `AMY_DATA_DIR` | `/data` | Persistent state directory. |
| `AMY_PORT` | `8484` | Listen port inside the container. |
| `AMY_HTTPS_PORT` | `8443` | The same app over HTTPS (`0` turns it off). Browsers only allow the microphone on HTTPS pages (or localhost). A self-signed certificate is made on first start in `/data/secrets/tls`; put your own `cert.pem` and `key.pem` there to use a real one. |
| `PUID` / `PGID` | `1000` / `1000` | Owner of files in `/data`. Unraid uses `99` / `100`. |
| `OLLAMA_MODELS` | `/data/ollama/models` | Where the bundled Ollama stores models. |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs the bundled Ollama can use. |
| `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE` | `/data/cache/cuda` / 4 GB | Persistent cache of compiled GPU kernels. Keeps the one-time first-run compile from repeating after restarts. |

Any other `OLLAMA_*` variable (e.g. `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`) is
passed through to the bundled Ollama.

## Examples

**Use a vLLM server and OpenAI alongside bundled Ollama:**

```env
AMY_PROVIDERS=[{"name":"vllm","base_url":"http://192.168.1.20:8000/v1"},{"name":"openai","base_url":"https://api.openai.com/v1","api_key":"sk-..."}]
```

**Slim image + existing Ollama on the host:**

```env
AMY_OLLAMA_URL=http://host.docker.internal:11434
```

## Rotating the API key

Set `AMY_API_KEY` to a new value, or delete `/data/secrets/api_key` and restart.
