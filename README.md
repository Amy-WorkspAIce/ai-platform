# Amy AI Platform

A self-hosted, customizable AI platform. It runs models on your own GPU, serves
them through an **OpenAI-compatible API**, and includes a web chat UI.

It is the first app of **[Amy WorkspAIce](https://github.com/Amy-WorkspAIce)**, an
open-source workplace suite (documents, spreadsheets, tasks, notes, code, chat, ...).
Each app is its own container. The AI Platform is the AI service your other apps
call, and it can act on those apps through their APIs.

> **Status: early development.** Working today: a ChatGPT-style chat, hardware-aware
> model discovery with one-click installs, a multi-provider OpenAI-compatible API,
> bundled Ollama with NVIDIA GPU support, and API-key auth. See [Roadmap](#roadmap).

## Features

- **One container, zero config.** Bundled [Ollama](https://ollama.com) with CUDA.
- **Model discovery for your hardware.** Amy detects your GPU, memory and disk, rates
  every model in a curated catalog ("Runs on GPU", "Partial GPU", "Too large"), and
  recommends the best fit for each use case: chat, reasoning, code, vision, embeddings.
  One click downloads it and pre-warms it for your GPU. Any other Ollama or Hugging Face
  GGUF model can be installed by name.
- **ChatGPT-style chat** with streaming, markdown, and collapsible "thinking" for reasoning models.
- **OpenAI-compatible API** at `/v1`. Point any OpenAI SDK or app at it.
- **Bring any engine.** Bundled Ollama, an external Ollama, llama.cpp server, vLLM,
  LM Studio, or hosted APIs. Models from all of them appear in one list, as
  `provider/model`.
- **Secure by default.** API key required, auto-generated on first start.
- **Runs anywhere.** Unraid template, Docker Compose, Docker Desktop (Windows/macOS), bare metal.

## Quick start

### Unraid

Install from Community Applications *(pending submission)*, or add the template
manually. See **[docs/install/unraid.md](docs/install/unraid.md)**.

### Docker (Linux, NVIDIA GPU)

```bash
docker run -d --name amy-ai-platform --gpus all \
  -p 8484:8484 -v ./amy-data:/data \
  ghcr.io/amy-workspaice/ai-platform:latest

docker logs amy-ai-platform          # shows your API key
```

Then open **http://localhost:8484**, paste the key, and go to **Models** to install a
model that fits your hardware. No GPU? Leave out `--gpus all`.

More options (compose, external Ollama, Windows/macOS):
[docs/install/docker.md](docs/install/docker.md) and [docs/install/other.md](docs/install/other.md).

### Use the API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8484/v1", api_key="amy-...")
reply = client.chat.completions.create(
    model="ollama/llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

## Image variants

| Tag | Contents | Use when |
|---|---|---|
| `latest` | Platform + bundled Ollama + CUDA runtime (large) | Plug-and-play, especially on Unraid |
| `slim` | Platform only (small) | You already run Ollama, vLLM, etc. elsewhere |

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [API](docs/api.md)
- [Connectors: how other apps plug in](docs/connectors.md)
- [Development](docs/development.md)

## Roadmap

- [x] OpenAI-compatible proxy over multiple providers
- [x] Web chat with streaming
- [x] Bundled Ollama + NVIDIA GPU, Unraid template
- [x] Hardware-aware model discovery, one-click install with progress, delete
- [ ] Persistent conversation history (sidebar chat list)
- [ ] Image generation and audio engines (catalog entries exist, marked "coming soon")
- [ ] Provider management in the UI
- [ ] Multiple API keys and users
- [ ] Customization: personas/system prompts, themes, plugins
- [ ] Connectors: use other Amy WorkspAIce apps (and any MCP server) as tools
- [ ] Embeddings API, RAG

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[AGPL-3.0-or-later](LICENSE). You may self-host, modify, and redistribute it. If you
offer a modified version to users over a network, you must publish your changes.
