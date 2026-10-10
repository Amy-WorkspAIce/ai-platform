# Install with Docker / Docker Compose (Linux)

## GPU prerequisites (optional)

Install the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then check it works with: `docker run --rm --gpus all ubuntu nvidia-smi`.

## Compose (recommended)

```bash
mkdir amy-ai && cd amy-ai
curl -fsSLO https://raw.githubusercontent.com/Amy-WorkspAIce/ai-platform/main/deploy/docker/docker-compose.yml
curl -fsSLO https://raw.githubusercontent.com/Amy-WorkspAIce/ai-platform/main/deploy/docker/docker-compose.gpu.yml

# CPU only:
docker compose up -d
# NVIDIA GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

docker compose logs ai-platform | grep "API key"
docker compose exec ai-platform ollama pull llama3.2
```

Open http://localhost:8484.

To configure it, create a `.env` next to the compose file. See
[configuration.md](../configuration.md) and `.env.example`.

## Separate Ollama container

If you prefer to run Ollama as its own container (or already have one), use
`deploy/docker/docker-compose.external-ollama.yml`. It runs the `slim` image and
sets `AMY_OLLAMA_URL=http://ollama:11434`.

## Build from source

```bash
git clone https://github.com/Amy-WorkspAIce/ai-platform && cd ai-platform
docker build -t amy-ai-platform .                 # full (bundled Ollama)
docker build -t amy-ai-platform:slim --target slim .
```

## Updating

```bash
docker compose pull && docker compose up -d
```

Your data and models persist in `./data`.
