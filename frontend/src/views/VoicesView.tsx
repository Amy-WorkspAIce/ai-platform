import { useEffect, useMemo, useRef, useState } from "react";
import type { Api, AppSettings, LibraryVoice, ModelVoices, SuggestedRunner, VoiceDetail, VoiceRunner } from "../api";
import { describeVoice, routeAudio, sampleText } from "../devices";
import { Icon } from "../icons";
import { CustomVoices, acceptsText } from "./CustomVoices";
import { FIT_LABEL } from "./ModelsView";
import { type Facet, type Sort, VoiceFilterBar, labelFacets, labelLine, useVoiceFilters } from "./VoiceFilters";

const YOURS = "Your voices";
const PAGE = 24;

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const LAST = "￿"; // sorts voices missing a label after the rest

function labelSorts<T>(label: (item: T) => VoiceDetail): Sort<T>[] {
  const name = (i: T) => label(i).name;
  return [
    { label: "Name", key: name },
    { label: "Gender", key: (i) => `${label(i).gender ?? LAST} ${name(i)}` },
    { label: "Language", key: (i) => `${label(i).language || LAST} ${name(i)}` },
    { label: "Accent", key: (i) => `${label(i).accent ?? LAST} ${name(i)}` },
    { label: "Age, youngest first", key: (i) => label(i).age ?? 999 },
  ];
}

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
          <VoiceLibrary api={api} version={version} onInstalled={() => setVersion((v) => v + 1)} onBrowseModels={onBrowseModels} />
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
    own.has(v)
      ? { ...own.get(v)!.meta, name: own.get(v)!.name, group: YOURS }
      : (info?.details?.[v] ?? describeVoice(v));
  const items = useMemo(() => [...(voices ?? []), ...own.keys()], [info]); // eslint-disable-line react-hooks/exhaustive-deps
  const facets: Facet<string>[] = [
    { key: "set", label: "Set", values: (v) => (label(v).group ? [label(v).group!] : []) },
    ...labelFacets(label),
  ];
  const filters = useVoiceFilters(items, facets, labelSorts(label), (v) => `${label(v).name} ${v} ${label(v).region ?? ""}`, model);
  const [shown, setShown] = useState(PAGE);
  useEffect(() => setShown(PAGE), [filters.matching.length, model]);
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

  async function choose(voice: string, on: string | null = model) {
    setError("");
    try {
      // A voice belongs to its model, so using another model's voice switches models too.
      const models = on === active ? settings.models : await api.saveModelDefault("speech", on);
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
              {items.length > 8 && <VoiceFilterBar state={filters} total={items.length} label="these voices" />}
              <div className="voice-grid">
                {filters.matching.slice(0, shown).map((v) => {
                  const d = label(v);
                  const sub = labelLine(d) || (own.has(v) ? (own.get(v)!.kind === "clone" ? "Sample" : "Described") : "");
                  return (
                    <div key={v} className={`voice ${v === current ? "selected" : ""}`}>
                      <button className="icon-button" title={`Hear ${d.name}`} disabled={!!playing} onClick={() => void preview(v)}>
                        {playing === v ? "…" : <Icon name="volume" size={14} label={`Hear ${d.name}`} />}
                      </button>
                      <span>
                        <strong>{d.name}</strong>
                        {(sub || d.region || d.language) && (
                          <span className="muted small voice-sub">{[sub, d.region ?? (d.accent ? "" : d.language)].filter(Boolean).join(" · ")}</span>
                        )}
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
              {items.length > 0 && filters.matching.length === 0 && <p className="muted small">No voices match.</p>}
              {filters.matching.length > shown && (
                <button className="secondary add-voice-button" onClick={() => setShown((n) => n + PAGE * 2)}>
                  Show more ({filters.matching.length - shown} left)
                </button>
              )}
            </>
          )}
        </>
      )}
      {error && <p className="error small">{error}</p>}

      <CustomVoices
        api={api}
        speech={speech}
        version={version}
        onChanged={loadVoices}
        active={active}
        current={settings.voice ?? null}
        onUse={(m, v) => void choose(v, m)}
        onBrowseModels={onBrowseModels}
      />
    </section>
  );
}

const NO_MODEL = "No installed model";

function VoiceLibrary({
  api,
  version,
  onInstalled,
  onBrowseModels,
}: {
  api: Api;
  version: number;
  onInstalled: () => void;
  onBrowseModels: () => void;
}) {
  const [voices, setVoices] = useState<LibraryVoice[] | null>(null);
  const [runners, setRunners] = useState<VoiceRunner[]>([]);
  const [suggested, setSuggested] = useState<SuggestedRunner[]>([]);
  const [shown, setShown] = useState(PAGE);
  const [playing, setPlaying] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    api
      .voiceLibrary()
      .then((r) => {
        setVoices(r.voices);
        setRunners(r.models);
        setSuggested(r.suggested);
      })
      .catch((e) => setError(errorText(e)));
    return () => audio.current?.pause();
  }, [api, version]);
  const runnerName = (id: string) => {
    const m = runners.find((r) => r.id === id);
    return m ? m.name + (m.how === "built_in" ? " (built in)" : "") : id;
  };

  const all = voices ?? [];
  const facets: Facet<LibraryVoice>[] = [
    { key: "collection", label: "Collection", values: (v) => [v.collection] },
    ...labelFacets((v: LibraryVoice) => v),
    { key: "runs-on", label: "Runs on", values: (v) => (v.runs_on.length ? v.runs_on.map(runnerName) : [NO_MODEL]) },
    { key: "license", label: "License", values: (v) => [v.license] },
    { key: "installed", label: "Installed", values: (v) => [v.installed ? "Installed" : "Not installed"] },
  ];
  const sorts: Sort<LibraryVoice>[] = [
    ...labelSorts((v: LibraryVoice) => v),
    { label: "Collection", key: (v) => `${v.collection} ${v.name}` },
    { label: "Installed first", key: (v) => `${v.installed ? 0 : 1} ${v.name}` },
  ];
  const filters = useVoiceFilters(all, facets, sorts, (v) => `${v.name} ${v.collection} ${v.region ?? ""} ${v.accent ?? ""}`);
  const matching = filters.matching;
  useEffect(() => setShown(PAGE), [matching.length]);

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
          {suggested.length > 0 && (
            <div className="runner-hint small">
              <p>
                {runners.length
                  ? "Some of these voices have no installed model that can speak them. To use them, install one of:"
                  : "To speak in these voices, install a speech model that can use them, such as:"}
              </p>
              <ul>
                {suggested.map((s) => (
                  <li key={s.tag}>
                    <strong>{s.name}</strong>
                    <span className="muted">
                      {" "}
                      · {s.note}
                      {s.fit ? ` · ${FIT_LABEL[s.fit]}` : ""}
                      {s.size_gb ? ` · ${s.size_gb} GB` : ""}
                    </span>
                  </li>
                ))}
              </ul>
              <button className="link small" onClick={onBrowseModels}>
                Browse models
              </button>
            </div>
          )}
          <VoiceFilterBar state={filters} total={all.length} label="the library" />
          <div className="voice-grid">
            {matching.slice(0, shown).map((v) => (
              <div key={v.id} className={`voice library-voice ${v.installed ? "selected" : ""}`}>
                <button className="icon-button" title={`Hear ${v.name}`} onClick={() => play(v)}>
                  {playing === v.id ? <Icon name="stop" size={14} label="Stop" /> : <Icon name="volume" size={14} label={`Hear ${v.name}`} />}
                </button>
                <span>
                  <strong>{v.name}</strong>
                  <span className="muted small voice-sub">{[labelLine(v), v.region, v.collection].filter(Boolean).join(" · ")}</span>
                </span>
                {v.installed ? (
                  <span className="tag tools">Installed</span>
                ) : (
                  <button className="link" disabled={!!busy} onClick={() => void install(v)}>
                    {busy === v.id ? "Installing…" : "Install"}
                  </button>
                )}
                <span className={`small runs-on ${v.runs_on.length ? "" : "none"}`}>
                  {v.runs_on.length ? `Runs on ${v.runs_on.map(runnerName).join(", ")}` : NO_MODEL}
                </span>
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
