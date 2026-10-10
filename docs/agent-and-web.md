# Agent mode and web access

Both features use **tool calling**: the model asks the platform to run a tool (search,
read a file, run a command), sees the result, and continues. You need a model that
supports tools. They're tagged **Tools** on the Models page (Qwen 3, gpt-oss, Qwen3-Coder,
Llama 3.x, Mistral Small). Bigger is better: an 8B model handles straightforward tasks
well, and a 1–2B model gets lost quickly.

## Web access

Turn on **🌐 Web** in the chat toolbar. The model gets two tools:

| Tool | What it does |
|---|---|
| `web_search` | Searches, then reads the top 3 results and includes excerpts, so answers come from page content rather than stale snippets. |
| `fetch_url` | Reads a page (or PDF) in full and returns its main text. |

### Search providers

| Provider | Setup | Notes |
|---|---|---|
| **SearXNG** (recommended) | Run SearXNG (Unraid: Community Apps → "SearXNG"; Docker: `deploy/docker/docker-compose.searxng.yml`) and set `AMY_SEARXNG_URL`. In SearXNG's `settings.yml`, add `json` under `search.formats`. | Self-hosted, private, aggregates many engines. |
| **Brave Search API** | Set `AMY_BRAVE_API_KEY` (free tier available). | Reliable; queries go to Brave. |
| **DuckDuckGo** | Nothing; the default fallback. | Scrapes DuckDuckGo's HTML page. Fine for personal use, but it can be rate-limited. |

`AMY_SEARCH_PROVIDER=auto` (default) picks SearXNG if configured, then Brave, then DuckDuckGo.

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
