# Other platforms

## Windows (Docker Desktop)

1. Install Docker Desktop with the WSL 2 backend.
2. **NVIDIA GPU:** install a current NVIDIA Windows driver. GPU passthrough to
   WSL 2 is built in. Check it with `docker run --rm --gpus all ubuntu nvidia-smi`.
3. Follow [docker.md](docker.md), running the commands in PowerShell.
   Use a Docker volume instead of `./data` if file performance is slow:
   `-v amy-data:/data`.

## macOS (Docker Desktop)

Containers on macOS cannot use Apple GPUs. You have two options:

- **Recommended:** run [Ollama natively](https://ollama.com/download) on the Mac
  (it uses Metal), and run the **slim** image pointed at it:
  ```bash
  docker run -d --name amy-ai-platform -p 8484:8484 -v amy-data:/data \
    -e AMY_OLLAMA_URL=http://host.docker.internal:11434 \
    ghcr.io/amy-workspaice/ai-platform:slim
  ```
- Run the full image CPU-only (slow for anything but tiny models).

## TrueNAS SCALE, Synology, CasaOS, Umbrel

Catalog templates are planned. Meanwhile, use the compose file from
[docker.md](docker.md) through each platform's "custom app / compose" feature.
Map `/data` to persistent storage, and expose port 8484.

## Kubernetes

A Helm chart is planned. The container needs a PVC at `/data`, port 8484, and
`nvidia.com/gpu` resources for the full image.

## Bare metal (no Docker)

See [development.md](../development.md). Run Ollama yourself, then run
`uvicorn app.main:create_app --factory` with `AMY_DATA_DIR` and `AMY_STATIC_DIR` set.
