import { useEffect, useState } from "react";
import { Api, type AppSettings } from "../api";

type Props = { api: Api; onChanged?: () => void };

export function SettingsView({ api, onChanged }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .settings()
      .then(setSettings)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  if (!settings) {
    return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  }

  return (
    <div className="page settings">
      <h1>Settings</h1>

      <HuggingFace
        api={api}
        status={settings.huggingface}
        onChange={(huggingface) => {
          setSettings({ ...settings, huggingface });
          onChanged?.();
        }}
      />

      <section className="card setting">
        <div className="setting-head">
          <h2>Web search</h2>
          <span className="tag tools">On</span>
        </div>
        <p className="muted small">
          Used when <strong>🌐 Web</strong> is on in a chat. Providers are tried in this order:
        </p>
        <ol className="providers">
          {settings.search.providers.map((p) => (
            <li key={p}>
              {p === "searxng"
                ? settings.search.bundled_searxng
                  ? "SearXNG (built in, private)"
                  : "SearXNG (your instance)"
                : p === "brave"
                  ? "Brave Search API"
                  : "DuckDuckGo (fallback)"}
            </li>
          ))}
        </ol>
        <p className="muted small">
          Change providers with environment variables (<code>AMY_SEARXNG_URL</code>, <code>AMY_BRAVE_API_KEY</code>).
        </p>
      </section>

      <section className="card setting">
        <h2>About</h2>
        <p className="muted small">
          Amy AI Platform v{settings.version} ·{" "}
          <a href="https://github.com/Amy-WorkspAIce/ai-platform" target="_blank" rel="noreferrer">
            GitHub
          </a>{" "}
          ·{" "}
          <a href="https://github.com/Amy-WorkspAIce/ai-platform/blob/main/docs/configuration.md" target="_blank" rel="noreferrer">
            Configuration reference
          </a>
        </p>
      </section>
    </div>
  );
}

function HuggingFace({
  api,
  status,
  onChange,
}: {
  api: Api;
  status: AppSettings["huggingface"];
  onChange: (s: AppSettings["huggingface"]) => void;
}) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function save() {
    setBusy(true);
    setError("");
    try {
      onChange(await api.saveHfToken(token.trim()));
      setToken("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm("Remove the Hugging Face token? Gated models you've installed keep working.")) return;
    setBusy(true);
    try {
      onChange(await api.clearHfToken());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card setting">
      <div className="setting-head">
        <h2>Hugging Face</h2>
        {status.token_set ? <span className="tag tools">Connected</span> : <span className="tag">Not connected</span>}
      </div>
      <p className="muted small">
        Some models (e.g. FLUX.1 [dev], Stable Diffusion 3.5) are <em>gated</em>: you accept their license on Hugging
        Face, and Amy downloads them with your token. Without one, they show <strong>Requires Hugging Face token</strong>{" "}
        instead of Install.
      </p>
      {status.token_set ? (
        <div className="hf-connected">
          <span>
            {status.username ? (
              <>
                Connected as <strong>@{status.username}</strong>
              </>
            ) : (
              "Token set"
            )}
            {status.source === "env" && <span className="muted small"> (from AMY_HF_TOKEN; edit it in the container settings)</span>}
          </span>
          {status.source !== "env" && (
            <button className="link danger" disabled={busy} onClick={remove}>
              Remove
            </button>
          )}
        </div>
      ) : (
        <form
          className="token-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (token.trim()) void save();
          }}
        >
          <input
            type="password"
            autoComplete="off"
            placeholder="hf_…"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
          <button className="primary" disabled={busy || !token.trim()}>
            {busy ? "Checking…" : "Save"}
          </button>
        </form>
      )}
      {error && <p className="error small">{error}</p>}
      <p className="muted small">
        Create a <strong>Read</strong> token at{" "}
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
          huggingface.co/settings/tokens
        </a>
        . It's stored on your server only (in <code>/data/secrets</code>) and used just for downloading models.
      </p>
    </section>
  );
}
