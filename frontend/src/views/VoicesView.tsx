import { useEffect, useMemo, useRef, useState } from "react";
import type { Api, AppSettings, LibraryVoice, ModelVoices, VoiceDetail } from "../api";
import { describeVoice, routeAudio, sampleText } from "../devices";
import { Icon } from "../icons";
import { CustomVoices, acceptsText } from "./CustomVoices";

const YOURS = "Your voices";
const PAGE = 24;

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * Everything about voices: browse and preview any installed speech model's voices and pick the
 * one Amy speaks with, manage your own voices, and install ready-made ones from the library.
 */
export function VoicesView({ api, onBrowseModels }: { api: Api; onBrowseModels: () => void }) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [error, setError] = useState("");
  // Bumped when the voice library adds a voice, so the lists that show your voices reload.
  const [version, setVersion] = useState(0);

  useEffect(() => {
    api
      .settings()
      .then(setSettings)
      .catch((e) => setError(errorText(e)));
  }, [api]);

  return (
    <div className="page settings voices-page">
      <h1>Voices</h1>
      {!settings ? (
        error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>
      ) : (
        <>
          <ModelVoicesCard api={api} settings={settings} version={version} onSettings={setSettings} onBrowseModels={onBrowseModels} />
          <VoiceLibrary api={api} onInstalled={() => setVersion((v) => v + 1)} />
        </>
      )}
    </div>
  );
}

function ModelVoicesCard({
  api,
  settings,
  version,
  onSettings,
  onBrowseModels,
}: {
  api: Api;
  settings: AppSettings;
  version: number;
  onSettings: (s: AppSettings) => void;
  onBrowseModels: () => void;
}) {
  const speech = settings.models.speech;
  const active = speech.pinned ?? speech.best;
  const [browsing, setBrowsing] = useState<string | null>(null);
  const model = browsing && speech.installed.some((m) => m.id === browsing) ? browsing : active;
  const [info, setInfo] = useState<ModelVoices | null>(null);
  const [lang, setLang] = useState("");
  const [playing, setPlaying] = useState("");
  const [error, setError] = useState("");

  const loadVoices = () => {
    if (!model) return;
    setInfo(null);
    setError("");
    api
      .voices(model)
      .then(setInfo)
      .catch((e) => setError(errorText(e)));
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(loadVoices, [api, model, version]);

  const voices = info?.voices ?? null;
  const own = new Map((info?.custom ?? []).map((v) => [v.id, v]));
  const inUse = model === active;
  const current = inUse ? (settings.voice ?? info?.voices[0] ?? "") : "";
  const label = (v: string): VoiceDetail =>
    own.has(v) ? { name: own.get(v)!.name, language: YOURS, gender: "" } : (info?.details?.[v] ?? describeVoice(v));
  const byLang = new Map<string, string[]>();
  for (const v of [...(voices ?? []), ...own.keys()]) {
    const l = label(v).language;
    byLang.set(l, [...(byLang.get(l) ?? []), v]);
  }
  const shownLang = byLang.has(lang) ? lang : byLang.has(label(current).language) ? label(current).language : [...byLang.keys()][0];
  const modelName = (id: string | null) => speech.installed.find((m) => m.id === id)?.name ?? id ?? "";

  async function preview(voice: string) {
    if (!model) return;
    setPlaying(voice);
    try {
      const text = own.has(voice)
        ? `Hi, I'm ${own.get(voice)!.name}, one of your voices.`
        : info?.details?.[voice]
          ? `Hi, I'm ${info.details[voice].name}, one of Amy's voices.`
          : sampleText(voice);
      const url = URL.createObjectURL(await api.speak(model, text, voice));
      const a = new Audio(url);
      routeAudio(a);
      a.onended = () => {
        setPlaying("");
        URL.revokeObjectURL(url);
      };
      await a.play();
    } catch (e) {
      setPlaying("");
      setError(errorText(e));
    }
  }

  async function choose(voice: string) {
    setError("");
    try {
      // A voice belongs to its model, so using another model's voice switches models too.
      const models = inUse ? settings.models : await api.saveModelDefault("speech", model);
      const saved = await api.saveVoice(voice);
      onSettings({ ...settings, models, voice: saved.voice });
      setBrowsing(null);
    } catch (e) {
      setError(errorText(e));
    }
  }

  return (
    <section className="card setting">
      <h2>Speaking voice</h2>
      {!model ? (
        <p className="muted small">
          Install a text-to-speech model to choose a voice.{" "}
          <button className="link" onClick={onBrowseModels}>
            Browse models
          </button>
        </p>
      ) : (
        <>
          <p className="muted small">
            Used for <strong>Read aloud</strong> and when chat speaks. Each speech model has its own voices: press play to
            hear one, and <strong>Use</strong> to speak with it.
          </p>
          {speech.installed.length > 1 && (
            <label className="model-default voices-model">
              <span>
                <strong>Model</strong>
                {!inUse && <span className="muted small"> · using a voice from it makes it your speech model</span>}
              </span>
              <select value={model} onChange={(e) => setBrowsing(e.target.value)}>
                {speech.installed.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                    {m.id === active ? " (in use)" : ""}
                  </option>
                ))}
              </select>
            </label>
          )}
          {speech.installed.length <= 1 && <p className="small">Model: {modelName(model)}</p>}
          {!voices ? (
            <p className="muted small">{error ? "" : "Loading voices…"}</p>
          ) : (
            <>
              <p className="muted small">
                {voices.length > 0 &&
                  `${voices.length} voice${voices.length === 1 ? "" : "s"}, nothing extra to download. `}
                {info && acceptsText(info)}
              </p>
              {byLang.size > 1 && (
                <div className="segmented wrap" role="radiogroup" aria-label="Language">
                  {[...byLang.keys()].map((l) => (
                    <button key={l} role="radio" aria-checked={shownLang === l} className={shownLang === l ? "active" : ""} onClick={() => setLang(l)}>
                      {l}
                    </button>
                  ))}
                </div>
              )}
              <div className="voice-grid">
                {(byLang.get(shownLang) ?? []).map((v) => {
                  const d = label(v);
                  return (
                    <div key={v} className={`voice ${v === current ? "selected" : ""}`}>
                      <button className="icon-button" title={`Hear ${d.name}`} disabled={!!playing} onClick={() => void preview(v)}>
                        {playing === v ? "…" : <Icon name="volume" size={14} label={`Hear ${d.name}`} />}
                      </button>
                      <span>
                        <strong>{d.name}</strong>
                        <span className="muted small">
                          {" "}
                          {d.gender || (own.has(v) ? (own.get(v)!.kind === "clone" ? "Sample" : "Described") : "")}
                        </span>
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
        </>
      )}
      {error && <p className="error small">{error}</p>}

      <CustomVoices
        api={api}
        speech={speech}
        usable={info ? new Set(own.keys()) : null}
        version={version}
        onChanged={loadVoices}
      />
    </section>
  );
}

function VoiceLibrary({ api, onInstalled }: { api: Api; onInstalled: () => void }) {
  const [voices, setVoices] = useState<LibraryVoice[] | null>(null);
  const [collection, setCollection] = useState("");
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const [playing, setPlaying] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    api
      .voiceLibrary()
      .then((r) => setVoices(r.voices))
      .catch((e) => setError(errorText(e)));
    return () => audio.current?.pause();
  }, [api]);

  const collections = useMemo(() => [...new Set((voices ?? []).map((v) => v.collection))], [voices]);
  const q = query.trim().toLowerCase();
  const matching = (voices ?? []).filter(
    (v) => (!collection || v.collection === collection) && (!q || `${v.name} ${v.collection} ${v.language}`.toLowerCase().includes(q)),
  );

  function play(v: LibraryVoice) {
    audio.current?.pause();
    if (playing === v.id) {
      setPlaying("");
      return;
    }
    const a = new Audio(v.preview);
    routeAudio(a);
    a.onended = () => setPlaying("");
    a.onerror = () => {
      setPlaying("");
      setError("Couldn't play that sample.");
    };
    audio.current = a;
    setPlaying(v.id);
    void a.play().catch(() => setPlaying(""));
  }

  async function install(v: LibraryVoice) {
    setBusy(v.id);
    setError("");
    try {
      const added = await api.installLibraryVoice(v.id);
      setVoices((all) => (all ?? []).map((x) => (x.id === v.id ? { ...x, installed: added.id } : x)));
      onInstalled();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="card setting">
      <h2>Voice library</h2>
      <p className="muted small">
        Ready-made voice samples that are free to use (public domain or CC BY), from Kyutai's voice library on Hugging Face.
        Install one and it joins your voices: every model that copies voices can speak in it.
      </p>
      {!voices ? (
        <p className="muted small">{error ? "" : "Loading the library…"}</p>
      ) : (
        <>
          <div className="library-filters">
            <div className="segmented wrap" role="radiogroup" aria-label="Collection">
              {["", ...collections].map((c) => (
                <button
                  key={c || "all"}
                  role="radio"
                  aria-checked={collection === c}
                  className={collection === c ? "active" : ""}
                  onClick={() => {
                    setCollection(c);
                    setShown(PAGE);
                  }}
                >
                  {c || "All"}
                </button>
              ))}
            </div>
            <input
              type="search"
              placeholder="Search voices"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setShown(PAGE);
              }}
              aria-label="Search voices"
            />
          </div>
          <div className="voice-grid">
            {matching.slice(0, shown).map((v) => (
              <div key={v.id} className={`voice ${v.installed ? "selected" : ""}`}>
                <button className="icon-button" title={`Hear ${v.name}`} onClick={() => play(v)}>
                  {playing === v.id ? <Icon name="stop" size={14} label="Stop" /> : <Icon name="volume" size={14} label={`Hear ${v.name}`} />}
                </button>
                <span>
                  <strong>{v.name}</strong>
                  <span className="muted small voice-sub">{collection ? v.license : `${v.collection} · ${v.license}`}</span>
                </span>
                {v.installed ? (
                  <span className="tag tools">Installed</span>
                ) : (
                  <button className="link" disabled={!!busy} onClick={() => void install(v)}>
                    {busy === v.id ? "Installing…" : "Install"}
                  </button>
                )}
              </div>
            ))}
          </div>
          {matching.length === 0 && <p className="muted small">No voices match.</p>}
          {matching.length > shown && (
            <button className="secondary add-voice-button" onClick={() => setShown((n) => n + PAGE * 2)}>
              Show more ({matching.length - shown} left)
            </button>
          )}
        </>
      )}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
