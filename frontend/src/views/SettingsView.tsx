import { useEffect, useState } from "react";
import { Api, getPref, setPref, type AppSettings } from "../api";
import {
  MIC_PREF,
  SPEAKER_PREF,
  canChooseSpeaker,
  describeVoice,
  routeAudio,
  sampleText,
  useAudioDevices,
  useDevicePref,
} from "../devices";
import { Icon } from "../icons";
import { canRecord } from "../recorder";

type Props = { api: Api; onChanged?: () => void; onSignOut: () => void };

type Theme = "dark" | "light" | "system";

function applyTheme(theme: Theme) {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
}

export function SettingsView({ api, onChanged, onSignOut }: Props) {
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

      <VoiceAndDevices api={api} settings={settings} onVoice={(voice) => setSettings({ ...settings, voice })} />

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
                  ? "SearXNG (built in, private; asks Bing, Yahoo, DuckDuckGo and others at once)"
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

      <Appearance />

      <section className="card setting">
        <h2>API key</h2>
        <p className="muted small">
          This browser remembers the API key you signed in with. Other apps and scripts use the same key to call the API
          (see <code>/data/secrets/api_key</code>). If you change the key on the server (<code>AMY_API_KEY</code>), sign
          out here and sign in with the new one.
        </p>
        <div>
          <button className="secondary" onClick={onSignOut}>
            Sign out of this browser
          </button>
        </div>
      </section>

      <section className="card setting">
        <h2>About</h2>
        <p className="muted small">
          Amy-WorkspAIce AI Platform v{settings.version} ·{" "}
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

function Appearance() {
  const [theme, setTheme] = useState<Theme>(() => (getPref("theme") as Theme) || "dark");
  const choose = (t: Theme) => {
    setTheme(t);
    setPref("theme", t);
    applyTheme(t);
  };
  return (
    <section className="card setting">
      <h2>Appearance</h2>
      <div className="segmented" role="radiogroup" aria-label="Theme">
        {(["dark", "light", "system"] as Theme[]).map((t) => (
          <button key={t} role="radio" aria-checked={theme === t} className={theme === t ? "active" : ""} onClick={() => choose(t)}>
            {t === "system" ? "Match system" : t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <p className="muted small">Dark is the default. This is saved in this browser only.</p>
    </section>
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

function VoiceAndDevices({
  api,
  settings,
  onVoice,
}: {
  api: Api;
  settings: AppSettings;
  onVoice: (voice: string | null) => void;
}) {
  const speech = settings.models.speech;
  const model = speech.pinned ?? speech.best;
  const [voices, setVoices] = useState<string[] | null>(null);
  const [lang, setLang] = useState("");
  const [playing, setPlaying] = useState("");
  const [error, setError] = useState("");
  const devices = useAudioDevices();
  const [mic, setMic] = useDevicePref(MIC_PREF);
  const [speaker, setSpeaker] = useDevicePref(SPEAKER_PREF);

  useEffect(() => {
    if (!model) return;
    api
      .voices(model)
      .then((r) => setVoices(r.voices))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api, model]);

  const current = settings.voice ?? "af_heart";
  const byLang = new Map<string, string[]>();
  for (const v of voices ?? []) {
    const l = describeVoice(v).language;
    byLang.set(l, [...(byLang.get(l) ?? []), v]);
  }
  const shownLang = lang || describeVoice(current).language;

  async function preview(voice: string) {
    if (!model) return;
    setPlaying(voice);
    try {
      const url = URL.createObjectURL(await api.speak(model, sampleText(voice), voice));
      const a = new Audio(url);
      routeAudio(a);
      a.onended = () => {
        setPlaying("");
        URL.revokeObjectURL(url);
      };
      await a.play();
    } catch (e) {
      setPlaying("");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function choose(voice: string) {
    setError("");
    try {
      onVoice((await api.saveVoice(voice)).voice);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <section className="card setting">
      <h2>Voices and devices</h2>
      <h3 className="setting-sub">Voice</h3>
      {!model ? (
        <p className="muted small">Install a text-to-speech model (such as Kokoro) from Models to choose a voice.</p>
      ) : !voices ? (
        <p className="muted small">{error ? "" : "Loading voices…"}</p>
      ) : (
        <>
          <p className="muted small">
            Used for <strong>Read aloud</strong> and when chat speaks. All {voices.length} voices come with the model, so
            there's nothing extra to download. Press play to hear one.
          </p>
          <div className="segmented wrap" role="radiogroup" aria-label="Language">
            {[...byLang.keys()].map((l) => (
              <button key={l} role="radio" aria-checked={shownLang === l} className={shownLang === l ? "active" : ""} onClick={() => setLang(l)}>
                {l}
              </button>
            ))}
          </div>
          <div className="voice-grid">
            {(byLang.get(shownLang) ?? []).map((v) => {
              const d = describeVoice(v);
              return (
                <div key={v} className={`voice ${v === current ? "selected" : ""}`}>
                  <button className="icon-button" title={`Hear ${d.name}`} disabled={!!playing} onClick={() => void preview(v)}>
                    {playing === v ? "…" : <Icon name="volume" size={14} label={`Hear ${d.name}`} />}
                  </button>
                  <span>
                    <strong>{d.name}</strong>
                    <span className="muted small"> {d.gender}</span>
                  </span>
                  {v === current ? (
                    <span className="tag tools">In use</span>
                  ) : (
                    <button className="link" onClick={() => void choose(v)}>
                      Use
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}

      <h3 className="setting-sub">Microphone and speakers</h3>
      {!devices.supported ? (
        <p className="muted small">
          This browser can't list audio devices here (it needs HTTPS), so it uses the system defaults.
        </p>
      ) : (
        <>
          <div className="model-defaults">
            <label className="model-default">
              <span>
                <strong>Microphone</strong>
                <span className="muted small"> · for dictation</span>
              </span>
              <select value={mic} onChange={(e) => setMic(e.target.value)}>
                <option value="">System default</option>
                {devices.mics
                  .filter((d) => d.deviceId && d.deviceId !== "default")
                  .map((d, i) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </option>
                  ))}
              </select>
            </label>
            <label className="model-default">
              <span>
                <strong>Speakers</strong>
                <span className="muted small"> · for read-aloud and audio</span>
              </span>
              {canChooseSpeaker() ? (
                <select value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
                  <option value="">System default</option>
                  {devices.speakers
                    .filter((d) => d.deviceId && d.deviceId !== "default")
                    .map((d, i) => (
                      <option key={d.deviceId} value={d.deviceId}>
                        {d.label || `Speakers ${i + 1}`}
                      </option>
                    ))}
                </select>
              ) : (
                <span className="muted small">This browser always uses the system output.</span>
              )}
            </label>
          </div>
          {!devices.labelled && canRecord() && (
            <p className="muted small">
              Browsers hide device names until you allow the microphone.{" "}
              <button className="link" onClick={() => devices.ask().catch(() => setError("Microphone access was blocked."))}>
                Allow and show names
              </button>
            </p>
          )}
          <p className="muted small">Saved in this browser only, since each device has its own hardware.</p>
        </>
      )}
      {!canRecord() && (
        <p className="muted small">
          The microphone only works on secure pages (HTTPS or localhost), and this page is plain HTTP. Open Amy through
          HTTPS (a reverse proxy or Tailscale) to dictate from this browser; attaching a recording works either way.
        </p>
      )}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
