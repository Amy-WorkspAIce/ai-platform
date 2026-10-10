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
| `AMY_UPSTREAM_TIMEOUT` | `600` | Seconds to wait on a model engine. |
| `AMY_DATA_DIR` | `/data` | Persistent state directory. |
| `AMY_PORT` | `8484` | Listen port inside the container. |
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
