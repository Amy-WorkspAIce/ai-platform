# Agent mode and web access

Both features use **tool calling**: the model asks the platform to run a tool (search,
read a file, run a command), sees the result, and continues. You need a model that
supports tools. They're tagged **Tools** on the Models page (Qwen 3, gpt-oss, Qwen3-Coder,
Llama 3.x, Mistral Small). Bigger is better: an 8B model handles straightforward tasks
well, and a 1–2B model gets lost quickly.

## Delegation: images, video, voice

Chat is the single interface. When a media model is installed, the chat model gets tools to
delegate to it:

| Tool | What it does |
|---|---|
| `generate_image` | Writes a detailed prompt and creates an image (shown inline, saved to the Library). |
| `edit_image` | Changes an existing image ("make it nighttime"). Uses true editing on models that accept an input image (FLUX.2 klein); otherwise regenerates with an adjusted prompt and the same seed. Works on images you attach too. |
| `generate_video` | Starts a background video job; a progress card in the chat turns into the clip. |
| `speak` | Text-to-speech (saved to the Library). |

Which model does the work: the one pinned in **Settings → Models**, otherwise the **best
installed** for your hardware (Perfect-match ranking). Asking for one by name ("use SDXL")
works too. Before a media job the chat model is unloaded from the GPU, and a fully-on-GPU
image pipeline is moved back to RAM afterwards, so the two take turns on the card.

Attach files with 📎: images become part of your message (so the model can edit them), audio
files are transcribed into it.

Models that can't call tools still chat normally; Amy notices and retries without tools.

## Web access

On by default (**Settings → Web access**). The model gets two tools and uses them when a
question needs current information:

| Tool | What it does |
|---|---|
| `web_search` | Searches, then reads the top 3 results and includes excerpts, so answers come from page content rather than stale snippets. |
| `fetch_url` | Reads a page (or PDF) in full and returns its main text. |

### Search providers

The **full image bundles SearXNG**, a private meta-search engine that runs inside the
container on `127.0.0.1:8888`. Nothing to set up. It queries several engines at once
(Google, Bing, DuckDuckGo, Brave, Wikipedia, ...) and keeps working when one of them
throttles your server, which a single-engine scraper can't do.

| Provider | Setup | Notes |
|---|---|---|
| **Bundled SearXNG** (default, full image) | None. Disable with `AMY_BUNDLED_SEARXNG=false`. | Private; secret generated per install in `/data/secrets`. |
| **Your own SearXNG** | Set `AMY_SEARXNG_URL` (with `json` enabled under `search.formats`). | For the slim image, or if you already run one (`deploy/docker/docker-compose.searxng.yml`). |
| **Brave Search API** | Set `AMY_BRAVE_API_KEY` (free tier available). | Reliable; queries go to Brave. |
| **DuckDuckGo** | Nothing. | Last-resort fallback. Scrapes one engine, so it gets rate-limited quickly. |

Providers are tried in that order and fall through on failure (including "all SearXNG
engines blocked"). If every provider fails, the model is told plainly that search is down,
and not to keep retrying, so it can use `fetch_url` on a known site or tell you.

### Safety

`fetch_url` refuses private and LAN addresses (your router, NAS UI, other containers) unless
`AMY_WEB_ALLOW_PRIVATE=true`. Web pages can contain instructions aimed at the model
("ignore previous instructions and…"); blocking the LAN limits what such a page can make
the model reach.

## Agent mode

Switch the toolbar to **Agent** and pick or create a **workspace**. Workspaces are folders in
`/data/workspaces/<name>`. Create an empty one, or clone a git repository.

Describe what you want ("add a dark mode toggle", "fix the failing test"), and the agent will:
look around, edit files, run commands (tests, builds, git), and report back.

| Tool | Approval |
|---|---|
| `list_files`, `read_file`, `search_files` | Never needed |
| `write_file`, `edit_file`, `run_command` | **Asked every time**, unless Auto-approve is on |

The approval card shows what will happen (the command, or the file content or diff). Choose
**Approve**, **Approve all in this run**, or **Deny** (the agent is told and asks how to proceed).

### The environment

- Commands run in a fresh, non-interactive `bash` in the workspace, with a 10-minute limit
  (`AMY_AGENT_COMMAND_TIMEOUT`).
- Tools available: `git`, `python3`, `node`/`npm`/`npx`, `ripgrep`, `curl`.
- If the workspace has a `.venv` (or `venv`), it's **automatically active**: after
  `python3 -m venv .venv`, plain `pip`/`python`/`pytest` use it.
- The platform's secrets (API key, provider keys, `AMY_*`) are removed from the command
  environment.
- The agent stops if it keeps repeating a failing action, and after 40 steps
  (`AMY_AGENT_MAX_STEPS`). Say "continue" to let it keep going.

### Git and GitHub

Set `AMY_GITHUB_TOKEN` to let the agent clone private repos and `git push`. The token is
handed to git through an in-memory credential helper and is never written to disk. Use a
**fine-grained token** scoped to only the repositories you want the agent to touch.
Commits are authored as "Amy Agent <amy-agent@localhost>" by default.

### Security model, honestly

The file tools can't leave the workspace, but **commands run as the container's user** and
can read anything that user can, including other workspaces and `/data`. That's why commands
need approval by default. Leave Auto-approve off unless you trust the task and the model,
and be especially careful combining Agent mode with Web access on untrusted sites.
A future version may run commands in a separate sandbox container.

## API

`POST /api/agent/chat` streams server-sent events. See [api.md](api.md).
