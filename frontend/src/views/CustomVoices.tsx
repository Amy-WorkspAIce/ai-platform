import { useEffect, useRef, useState } from "react";
import type { Api, CustomVoice, ModelChoice, ModelVoices } from "../api";
import { describeVoice, routeAudio, toWav } from "../devices";
import { Icon } from "../icons";
import { canRecord, useRecorder } from "../recorder";

/** Read aloud when recording a sample: varied sounds, about 15 seconds at an easy pace. */
export const READ_ALOUD =
  "The quick brown fox jumps over the lazy dog. I love a good story, a warm cup of tea, and a quiet " +
  "evening at home. Is it going to rain tomorrow? I hope not, because we have plans to walk by the sea.";

const DESIGN_IDEAS = [
  "A calm older man with a deep, gravelly voice",
  "A cheerful young woman who speaks quickly",
  "A soft-spoken storyteller with a gentle British accent",
  "A confident newsreader, clear and neutral",
];

type Mode = "record" | "upload" | "describe" | "builtin";

const MODES: { id: Mode; label: string }[] = [
  { id: "record", label: "Record" },
  { id: "upload", label: "Upload" },
  { id: "describe", label: "Describe" },
  { id: "builtin", label: "From a built-in voice" },
];

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** What a model can do with custom voices, in a sentence. */
export function acceptsText(info: ModelVoices): string {
  const clone = info.accepts.includes("clone");
  const design = info.accepts.includes("design");
  if (!clone && !design) return "This model only uses its own voices.";
  if (!info.voices.length && clone) return "This model only speaks in a voice copied from a sample, so add one below.";
  if (clone && design) return "This model can also use voices you describe and voices copied from a sample.";
  if (design) return "This model can also use voices you describe in words.";
  return info.needs_text
    ? "This model can also copy a voice from a sample, if the sample has its words written down."
    : "This model can also copy a voice from a sample.";
}

/**
 * The user's own voices (shared by every model that accepts them) and the form to add one:
 * record yourself, upload a clip, describe a voice, or capture one of a model's built-in voices.
 */
export function CustomVoices({
  api,
  speech,
  usable,
  onChanged,
}: {
  api: Api;
  speech: ModelChoice;
  /** Ids of the voices the current model can use, or null if unknown. */
  usable: Set<string> | null;
  onChanged: () => void;
}) {
  const [voices, setVoices] = useState<CustomVoice[] | null>(null);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState("");

  const load = () =>
    api
      .customVoices()
      .then(setVoices)
      .catch((e) => setError(errorText(e)));
  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  function changed() {
    void load();
    onChanged();
  }

  return (
    <>
      <h3 className="setting-sub">Your voices</h3>
      <p className="muted small">
        Voices you add here work with every model that can use them: copy a voice from a short recording, or describe one
        in words. Only use recordings of your own voice, or of someone who has agreed.
      </p>
      {voices && voices.length > 0 && (
        <ul className="custom-voices">
          {voices.map((v) => (
            <VoiceRow key={v.id} api={api} voice={v} usable={usable} onChanged={changed} onError={setError} />
          ))}
        </ul>
      )}
      {adding ? (
        <AddVoice
          api={api}
          speech={speech}
          onDone={(added) => {
            setAdding(false);
            if (added) changed();
          }}
        />
      ) : (
        <button className="secondary add-voice-button" onClick={() => setAdding(true)}>
          Add a voice
        </button>
      )}
      {error && <p className="error small">{error}</p>}
    </>
  );
}

function VoiceRow({
  api,
  voice,
  usable,
  onChanged,
  onError,
}: {
  api: Api;
  voice: CustomVoice;
  usable: Set<string> | null;
  onChanged: () => void;
  onError: (e: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(voice.name);
  const [words, setWords] = useState(voice.kind === "clone" ? voice.transcript ?? "" : voice.description ?? "");
  const [playing, setPlaying] = useState(false);

  async function play() {
    setPlaying(true);
    try {
      const url = await api.voiceSampleUrl(voice.id);
      const a = new Audio(url);
      routeAudio(a);
      a.onended = () => {
        setPlaying(false);
        URL.revokeObjectURL(url);
      };
      await a.play();
    } catch (e) {
      setPlaying(false);
      onError(errorText(e));
    }
  }

  async function save() {
    try {
      await api.updateVoice(
        voice.id,
        voice.kind === "clone" ? { name, transcript: words } : { name, description: words },
      );
      setEditing(false);
      onChanged();
    } catch (e) {
      onError(errorText(e));
    }
  }

  async function remove() {
    try {
      await api.deleteVoice(voice.id);
      onChanged();
    } catch (e) {
      onError(errorText(e));
    }
  }

  return (
    <li className="custom-voice">
      <div className="custom-voice-head">
        {voice.kind === "clone" ? (
          <button className="icon-button" title="Hear the sample" disabled={playing} onClick={() => void play()}>
            {playing ? "…" : <Icon name="volume" size={14} label="Hear the sample" />}
          </button>
        ) : (
          <span className="icon-button placeholder" aria-hidden />
        )}
        <span className="custom-voice-name">
          <strong>{voice.name}</strong>
          <span className="muted small">
            {" "}
            · {voice.kind === "clone" ? `${voice.seconds ?? "?"} s sample` : "Described"}
          </span>
        </span>
        {usable && !usable.has(voice.id) && <span className="tag">Not for this model</span>}
        <button className="link" onClick={() => setEditing(!editing)}>
          {editing ? "Cancel" : "Edit"}
        </button>
        <button className="link danger" onClick={() => void remove()}>
          Delete
        </button>
      </div>
      {editing ? (
        <div className="custom-voice-edit">
          <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Name" maxLength={60} />
          <textarea
            value={words}
            rows={2}
            onChange={(e) => setWords(e.target.value)}
            aria-label={voice.kind === "clone" ? "What the sample says" : "Description"}
            placeholder={voice.kind === "clone" ? "What the sample says, word for word" : "Describe the voice"}
          />
          <button onClick={() => void save()}>Save</button>
        </div>
      ) : (
        <p className="muted small custom-voice-words">
          {voice.kind === "design"
            ? voice.description
            : voice.transcript
              ? `"${voice.transcript}"`
              : "No words written down (some models need them: edit to add what the sample says)."}
        </p>
      )}
    </li>
  );
}

function AddVoice({ api, speech, onDone }: { api: Api; speech: ModelChoice; onDone: (added: boolean) => void }) {
  const [mode, setMode] = useState<Mode>(canRecord() ? "record" : "upload");
  const [name, setName] = useState("");
  const [sample, setSample] = useState<{ wav: Blob; seconds: number; url: string } | null>(null);
  const [words, setWords] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const recorder = useRecorder();
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<number | null>(null);
  // From a built-in voice: which installed model and voice.
  const [source, setSource] = useState(speech.pinned ?? speech.best ?? "");
  const [sourceVoices, setSourceVoices] = useState<string[]>([]);
  const [sourceVoice, setSourceVoice] = useState("");

  useEffect(() => () => {
    if (sample) URL.revokeObjectURL(sample.url);
  }, [sample]);

  useEffect(() => {
    if (mode !== "builtin" || !source) return;
    setSourceVoices([]);
    api
      .voices(source)
      .then((r) => {
        setSourceVoices(r.voices);
        setSourceVoice(r.voices[0] ?? "");
      })
      .catch((e) => setError(errorText(e)));
  }, [api, mode, source]);

  async function useAudio(blob: Blob) {
    setError("");
    try {
      const { wav, seconds, trimmed } = await toWav(blob);
      if (seconds < 3) throw new Error("That's too short: the sample needs at least 3 seconds.");
      setSample({ wav, seconds, url: URL.createObjectURL(wav) });
      setNote(trimmed ? "Only the first 30 seconds are kept." : "");
    } catch (e) {
      setError(e instanceof DOMException ? "This browser can't read that audio file." : errorText(e));
    }
  }

  async function toggleRecording() {
    if (recorder.recording) {
      if (timer.current) window.clearInterval(timer.current);
      const blob = await recorder.stop();
      if (blob) await useAudio(blob);
      return;
    }
    setError("");
    try {
      await recorder.start();
      setWords(READ_ALOUD);
      setElapsed(0);
      const started = Date.now();
      timer.current = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 250);
    } catch {
      setError("Microphone access was blocked.");
    }
  }

  useEffect(() => () => void (timer.current && window.clearInterval(timer.current)), []);

  async function save() {
    setBusy(true);
    setError("");
    try {
      if (mode === "describe") {
        await api.addVoice({ name: name.trim(), kind: "design", description: description.trim() });
      } else if (mode === "builtin") {
        await api.voiceFromSpeech(name.trim(), source, sourceVoice);
      } else if (sample) {
        // Without words, the platform transcribes it; the list shows the result.
        await api.addVoice({ name: name.trim(), kind: "clone", transcript: words.trim(), sample: sample.wav });
      }
      onDone(true);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const ready =
    !!name.trim() &&
    !busy &&
    (mode === "describe" ? !!description.trim() : mode === "builtin" ? !!source && !!sourceVoice : !!sample);

  return (
    <div className="add-voice">
      <div className="segmented wrap" role="radiogroup" aria-label="How to add the voice">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="radio"
            aria-checked={mode === m.id}
            className={mode === m.id ? "active" : ""}
            onClick={() => {
              setMode(m.id);
              setError("");
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      <label className="field">
        <span>Name</span>
        <input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder="e.g. My voice, Captain Reed" />
      </label>

      {mode === "record" &&
        (canRecord() ? (
          <>
            <p className="muted small">
              Press record and read this aloud in a quiet room (10 to 20 seconds is plenty):
            </p>
            <blockquote className="read-aloud">{READ_ALOUD}</blockquote>
            <button className={recorder.recording ? "danger" : "secondary"} onClick={() => void toggleRecording()}>
              <Icon name={recorder.recording ? "stop" : "mic"} size={14} />{" "}
              {recorder.recording ? `Stop (${elapsed} s)` : sample ? "Record again" : "Record"}
            </button>
          </>
        ) : (
          <p className="muted small">
            Recording needs a secure page (HTTPS or localhost). Record on your phone or computer and use Upload instead.
          </p>
        ))}

      {mode === "upload" && (
        <label className="field">
          <span>Audio file (3 to 30 seconds of one person speaking; longer files are cut)</span>
          <input type="file" accept="audio/*" onChange={(e) => e.target.files?.[0] && void useAudio(e.target.files[0])} />
        </label>
      )}

      {(mode === "record" || mode === "upload") && sample && (
        <>
          <audio controls src={sample.url} ref={routeAudio} />
          <p className="muted small">
            {sample.seconds.toFixed(1)} seconds. {note}
          </p>
          <label className="field">
            <span>What the sample says, word for word (optional, but it makes the copy closer)</span>
            <textarea rows={3} value={words} onChange={(e) => setWords(e.target.value)} />
          </label>
          {!words.trim() && (
            <p className="muted small">Left blank, Amy writes it down with your speech-to-text model if one is installed.</p>
          )}
        </>
      )}

      {mode === "describe" && (
        <>
          <label className="field">
            <span>Describe the voice: who is speaking, their age, tone, pace and accent</span>
            <textarea rows={3} maxLength={500} value={description} onChange={(e) => setDescription(e.target.value)} />
          </label>
          <div className="chips">
            {DESIGN_IDEAS.map((idea) => (
              <button key={idea} className="chip" onClick={() => setDescription(idea)}>
                {idea}
              </button>
            ))}
          </div>
          <p className="muted small">
            Used by models that make voices from a description (Qwen3-TTS VoiceDesign, OmniVoice, VoxCPM). The same
            description can sound a little different each time.
          </p>
        </>
      )}

      {mode === "builtin" &&
        (speech.installed.length === 0 ? (
          <p className="muted small">Install a text-to-speech model first.</p>
        ) : (
          <>
            <p className="muted small">
              Amy speaks a sentence in the voice you pick and saves it as a sample, so models that copy voices (such as
              Qwen3-TTS Base) can use it.
            </p>
            <div className="model-defaults">
              <label className="model-default">
                <span>
                  <strong>Model</strong>
                </span>
                <select value={source} onChange={(e) => setSource(e.target.value)}>
                  {speech.installed.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="model-default">
                <span>
                  <strong>Voice</strong>
                </span>
                <select value={sourceVoice} onChange={(e) => setSourceVoice(e.target.value)} disabled={!sourceVoices.length}>
                  {sourceVoices.map((v) => (
                    <option key={v} value={v}>
                      {describeVoice(v).name}
                      {describeVoice(v).language !== "Voices" ? ` (${describeVoice(v).language})` : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {source && !sourceVoices.length && !error && (
              <p className="muted small">This model has no built-in voices to copy; pick another.</p>
            )}
          </>
        ))}

      <div className="row-actions">
        <button disabled={!ready} onClick={() => void save()}>
          {busy ? (mode === "builtin" ? "Speaking…" : "Saving…") : "Save voice"}
        </button>
        <button className="link" onClick={() => onDone(false)}>
          Cancel
        </button>
      </div>
      {error && <p className="error small">{error}</p>}
    </div>
  );
}
