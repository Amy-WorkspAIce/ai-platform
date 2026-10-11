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

      <DefaultModels
        api={api}
        models={settings.models}
        onChange={(models) => setSettings({ ...settings, models })}
      />

      <section className="card setting">
        <div className="setting-head">
          <h2>Web access</h2>
          <label className="switch">
            <input
              type="checkbox"
              checked={settings.web_access}
              onChange={async (e) => {
                const enabled = e.target.checked;
                try {
                  await api.saveWebAccess(enabled);
                  setSettings({ ...settings, web_access: enabled });
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err));
                }
              }}
            />
            {settings.web_access ? "On" : "Off"}
          </label>
        </div>
        <p className="muted small">
          When on, chat models can search the web and read pages whenever a question needs current information.
          Searches are tried with these providers, in order:
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


const CAPABILITY_LABEL: Record<string, { title: string; hint: string }> = {
  image: { title: "Images", hint: "Used when you ask chat to create or edit an image." },
  video: { title: "Video", hint: "Used when you ask chat for a video clip." },
  speech: { title: "Voice (text to speech)", hint: "Used for read-aloud and when chat speaks." },
  transcription: { title: "Transcription (speech to text)", hint: "Used for dictation and attached audio." },
};

function DefaultModels({
  api,
  models,
  onChange,
}: {
  api: Api;
  models: AppSettings["models"];
  onChange: (m: AppSettings["models"]) => void;
}) {
  const [error, setError] = useState("");

  async function choose(capability: string, value: string) {
    setError("");
    try {
      onChange(await api.saveModelDefault(capability, value || null));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <section className="card setting">
      <h2>Models</h2>
      <p className="muted small">
        Chat hands work off to these models. <strong>Best installed</strong> picks the most capable installed model that
        runs well on your hardware, so you don't need to change anything here unless you want a specific one.
      </p>
      <div className="model-defaults">
        {(Object.keys(CAPABILITY_LABEL) as (keyof AppSettings["models"])[]).map((cap) => {
          const choice = models[cap];
          const best = choice.installed.find((m) => m.id === choice.best);
          return (
            <label key={cap} className="model-default">
              <span>
                <strong>{CAPABILITY_LABEL[cap].title}</strong>
                <span className="muted small"> · {CAPABILITY_LABEL[cap].hint}</span>
              </span>
              {choice.installed.length === 0 ? (
                <span className="muted small">None installed (install one from Models)</span>
              ) : (
                <select value={choice.pinned ?? ""} onChange={(e) => void choose(cap, e.target.value)}>
                  <option value="">Best installed{best ? ` (${best.name})` : ""}</option>
                  {choice.installed.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              )}
            </label>
          );
        })}
      </div>
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
