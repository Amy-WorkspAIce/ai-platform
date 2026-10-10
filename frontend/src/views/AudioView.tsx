import { useEffect, useRef, useState } from "react";
import { Api, type ModelInfo } from "../api";
import { canRecord, extensionFor, useRecorder } from "../recorder";

type Props = {
  api: Api;
  sttModels: ModelInfo[] | null;
  ttsModels: ModelInfo[] | null;
  onBrowseModels: () => void;
};

export function AudioView({ api, sttModels, ttsModels, onBrowseModels }: Props) {
  return (
    <div className="page">
      <h1>Audio</h1>
      <div className="audio-grid">
        <Transcribe api={api} models={sttModels} onBrowseModels={onBrowseModels} />
        <Speak api={api} models={ttsModels} onBrowseModels={onBrowseModels} />
      </div>
    </div>
  );
}

function ModelPicker({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (v: string) => void }) {
  if (models.length < 2) return null;
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.name ?? m.id}
        </option>
      ))}
    </select>
  );
}

function Missing({ what, onBrowseModels }: { what: string; onBrowseModels: () => void }) {
  return (
    <>
      <p className="muted">Install a {what} model first. Amy will set up the audio engine for you.</p>
      <button className="primary" onClick={onBrowseModels}>
        Choose a model
      </button>
    </>
  );
}

function Transcribe({ api, models, onBrowseModels }: { api: Api; models: ModelInfo[] | null; onBrowseModels: () => void }) {
  const [model, setModel] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { recording, start, stop } = useRecorder();
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (models?.length && !models.some((m) => m.id === model)) setModel(models[0].id);
  }, [models, model]);

  async function run(audio: Blob, name: string) {
    setBusy(true);
    setError("");
    try {
      setText(await api.transcribe(model, audio, name));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecording() {
    if (!recording) {
      try {
        await start();
      } catch {
        setError("Couldn't access the microphone.");
      }
      return;
    }
    const blob = await stop();
    if (blob) await run(blob, `recording.${extensionFor(blob)}`);
  }

  return (
    <section className="card audio-card">
      <h2 className="small-heading">Speech to text</h2>
      {models !== null && models.length === 0 ? (
        <Missing what="speech-to-text" onBrowseModels={onBrowseModels} />
      ) : (
        <>
          <ModelPicker models={models ?? []} value={model} onChange={setModel} />
          <div className="row-actions start">
            <button className="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
              Upload audio
            </button>
            <button
              className={`secondary ${recording ? "recording" : ""}`}
              disabled={busy || !canRecord()}
              onClick={toggleRecording}
              title={canRecord() ? "" : "Microphone access needs HTTPS (or localhost)"}
            >
              {recording ? "● Stop recording" : "🎤 Record"}
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="audio/*,video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void run(f, f.name);
                e.target.value = "";
              }}
            />
          </div>
          {!canRecord() && (
            <p className="muted small">
              Recording needs HTTPS or localhost. Uploading files works everywhere.
            </p>
          )}
          {busy && <p className="muted">Transcribing… the first run loads the model.</p>}
          {error && <p className="error">{error}</p>}
          {text && (
            <>
              <textarea className="transcript" value={text} onChange={(e) => setText(e.target.value)} />
              <div className="row-actions start">
                <button className="secondary" onClick={() => navigator.clipboard?.writeText(text)}>
                  Copy
                </button>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}

function Speak({ api, models, onBrowseModels }: { api: Api; models: ModelInfo[] | null; onBrowseModels: () => void }) {
  const [model, setModel] = useState("");
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1);
  const [text, setText] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (models?.length && !models.some((m) => m.id === model)) setModel(models[0].id);
  }, [models, model]);

  useEffect(() => {
    if (!model) return;
    api
      .voices(model)
      .then((v) => {
        setVoices(v.voices);
        if (v.voices.length && !v.voices.includes(voice)) setVoice(v.voices[0]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api, model]);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  async function run() {
    if (!text.trim()) return;
    setBusy(true);
    setError("");
    try {
      setAudioUrl(URL.createObjectURL(await api.speak(model, text.trim(), voice, speed)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card audio-card">
      <h2 className="small-heading">Text to speech</h2>
      {models !== null && models.length === 0 ? (
        <Missing what="text-to-speech" onBrowseModels={onBrowseModels} />
      ) : (
        <>
          <ModelPicker models={models ?? []} value={model} onChange={setModel} />
          <textarea
            className="transcript"
            value={text}
            placeholder="Type something for Amy to say…"
            onChange={(e) => setText(e.target.value)}
          />
          <div className="row-actions start wrap">
            <select value={voice} onChange={(e) => setVoice(e.target.value)} aria-label="Voice">
              {voices.map((v) => (
                <option key={v} value={v}>
                  {voiceLabel(v)}
                </option>
              ))}
            </select>
            <label className="speed">
              Speed {speed.toFixed(1)}×
              <input type="range" min={0.5} max={2} step={0.1} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
            </label>
            <button className="primary" disabled={busy || !text.trim()} onClick={run}>
              {busy ? "Generating…" : "Speak"}
            </button>
          </div>
          {error && <p className="error">{error}</p>}
          {audioUrl && (
            <div className="audio-result">
              <audio src={audioUrl} controls autoPlay />
              <a className="button secondary" href={audioUrl} download="amy-speech.wav">
                Download
              </a>
            </div>
          )}
        </>
      )}
    </section>
  );
}

const ACCENTS: Record<string, string> = {
  a: "American",
  b: "British",
  e: "Spanish",
  f: "French",
  h: "Hindi",
  i: "Italian",
  j: "Japanese",
  p: "Portuguese",
  z: "Chinese",
};

/** "af_heart" -> "Heart (American, female)" */
function voiceLabel(id: string): string {
  const [prefix, name = id] = id.split("_");
  const accent = ACCENTS[prefix?.[0]] ?? "";
  const gender = prefix?.[1] === "f" ? "female" : prefix?.[1] === "m" ? "male" : "";
  const details = [accent, gender].filter(Boolean).join(", ");
  return `${name[0].toUpperCase()}${name.slice(1)}${details ? ` (${details})` : ""}`;
}
