import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, getPref, setPref, type GalleryVideo, type ModelInfo, type VideoJob } from "../api";
import { useBlobUrls } from "../blobs";

type Props = { api: Api; models: ModelInfo[] | null; onBrowseModels: () => void };

const SHAPES = [
  { id: "landscape", label: "Landscape", w: 16, h: 9 },
  { id: "portrait", label: "Portrait", w: 9, h: 16 },
  { id: "square", label: "Square", w: 1, h: 1 },
];
const DURATIONS = [2, 3, 5];

/** Keep the model's native pixel count for the chosen aspect ratio (multiples of 32). */
function dims(shape: (typeof SHAPES)[number], baseW: number, baseH: number) {
  const area = baseW * baseH;
  const scale = Math.sqrt(area / (shape.w * shape.h));
  const r = (x: number) => Math.max(256, Math.round((x * scale) / 32) * 32);
  return { width: r(shape.w), height: r(shape.h) };
}

const PHASES: Record<string, string> = {
  queued: "Waiting",
  optimizing: "Optimizing model files (first run only)",
  loading: "Loading the model (first run takes a while)",
  generating: "Generating",
  encoding: "Encoding video",
};

export function VideoView({ api, models, onBrowseModels }: Props) {
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [shape, setShape] = useState("landscape");
  const [seconds, setSeconds] = useState(2);
  const [steps, setSteps] = useState("");
  const [seed, setSeed] = useState("");
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [videos, setVideos] = useState<GalleryVideo[]>([]);
  const [error, setError] = useState("");
  const [now, setNow] = useState(Date.now());
  const info = models?.find((m) => m.id === model);
  const fetchUrl = useCallback((id: string) => api.videoUrl(id), [api]);
  const ids = useMemo(() => videos.map((v) => v.id), [videos]);
  const urls = useBlobUrls(fetchUrl, ids);

  useEffect(() => {
    if (!models?.length || models.some((m) => m.id === model)) return;
    const last = getPref("videoModel");
    setModel(models.some((m) => m.id === last) ? last : models[0].id);
  }, [models, model]);

  const refresh = useCallback(async () => {
    try {
      const [j, v] = await Promise.all([api.videoJobs(), api.videos()]);
      setJobs(j);
      setVideos(v);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => {
      setNow(Date.now());
      void refresh();
    }, 2000);
    return () => clearInterval(t);
  }, [active, refresh]);

  async function generate() {
    if (!prompt.trim() || !model) return;
    setError("");
    const fps = info?.fps ?? 16;
    const base = { w: info?.default_width ?? 832, h: info?.default_height ?? 480 };
    const { width, height } = dims(SHAPES.find((s) => s.id === shape)!, base.w, base.h);
    try {
      await api.generateVideo({
        model,
        prompt: prompt.trim(),
        negative_prompt: info?.negative_prompt && negative.trim() ? negative.trim() : undefined,
        width,
        height,
        frames: seconds * fps + 1,
        fps,
        steps: steps ? Number(steps) : undefined,
        seed: seed ? Number(seed) : undefined,
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove(v: GalleryVideo) {
    if (!confirm("Delete this video?")) return;
    try {
      await api.deleteVideo(v.id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (models !== null && models.length === 0) {
    return (
      <div className="page">
        <h1>Video</h1>
        <div className="welcome-card left">
          <p>
            Create short video clips from a text description. Install a video model first (Wan 2.1 1.3B is made for
            consumer GPUs). Amy sets up the engine for you.
          </p>
          <button className="primary" onClick={onBrowseModels}>
            Choose a video model
          </button>
        </div>
      </div>
    );
  }

  const visibleJobs = jobs.filter((j) => j.status !== "done");

  return (
    <div className="page">
      <h1>Video</h1>
      <form
        className="card image-form"
        onSubmit={(e) => {
          e.preventDefault();
          void generate();
        }}
      >
        <textarea
          className="prompt"
          value={prompt}
          placeholder="Describe the clip… e.g. a red fox trotting through fresh snow at dawn, cinematic, slow camera pan"
          onChange={(e) => setPrompt(e.target.value)}
        />
        {info?.negative_prompt && (
          <input value={negative} placeholder="Avoid (optional): blurry, distorted, static" onChange={(e) => setNegative(e.target.value)} />
        )}
        <div className="image-controls">
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setPref("videoModel", e.target.value);
            }}
          >
            {models?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name ?? m.id}
              </option>
            ))}
          </select>
          <div className="segmented">
            {SHAPES.map((s) => (
              <button key={s.id} type="button" className={shape === s.id ? "active" : ""} onClick={() => setShape(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
          <div className="segmented">
            {DURATIONS.map((d) => (
              <button key={d} type="button" className={seconds === d ? "active" : ""} onClick={() => setSeconds(d)}>
                {d}s
              </button>
            ))}
          </div>
          <details className="advanced">
            <summary>Advanced</summary>
            <label>
              Steps <input type="number" min={1} max={100} value={steps} placeholder="default" onChange={(e) => setSteps(e.target.value)} />
            </label>
            <label>
              Seed <input type="number" value={seed} placeholder="random" onChange={(e) => setSeed(e.target.value)} />
            </label>
          </details>
          <button type="submit" className="primary generate" disabled={!prompt.trim() || !model}>
            Generate
          </button>
        </div>
        <p className="muted small">
          Video is slow: a few seconds of footage can take several minutes. You can leave this page; the clip keeps
          rendering and lands in your gallery.
        </p>
        {error && <p className="error">{error}</p>}
      </form>

      {visibleJobs.length > 0 && (
        <section className="card">
          {visibleJobs.map((j) => (
            <div key={j.id} className="pull">
              <div className="pull-head">
                <span className="clip-prompt">{j.prompt}</span>
                <span className={j.status === "error" ? "error small" : "muted small"}>
                  {j.status === "error"
                    ? j.error
                    : `${PHASES[j.phase] ?? j.phase}${j.phase === "generating" && j.steps ? ` · step ${j.step}/${j.steps}` : ""} · ${Math.round((now - j.started_at * 1000) / 1000)}s`}
                </span>
              </div>
              {j.status !== "error" && (
                <div className="progress">
                  <div
                    style={{
                      width: `${j.phase === "encoding" ? 100 : j.steps ? Math.round((j.step / j.steps) * 100) : 3}%`,
                    }}
                  />
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      <div className="video-gallery">
        {videos.map((v) => (
          <figure key={v.id} className="card video-card">
            {urls[v.id] ? <video src={urls[v.id]} controls loop playsInline preload="metadata" /> : <div className="video-placeholder muted">…</div>}
            <figcaption>
              <p className="clip-prompt">{v.prompt}</p>
              <p className="muted small">
                {v.model.replace("image/", "").split("/").pop()} · {v.params.width}×{v.params.height} ·{" "}
                {(v.params.frames / v.params.fps).toFixed(1)}s
                {v.params.seconds ? ` · rendered in ${Math.round(v.params.seconds)}s` : ""}
              </p>
              <div className="row-actions start">
                {urls[v.id] && (
                  <a className="button secondary" href={urls[v.id]} download={`amy-${v.id}.mp4`}>
                    Download
                  </a>
                )}
                <button className="link" onClick={() => setPrompt(v.prompt)}>
                  Reuse prompt
                </button>
                <button className="link danger" onClick={() => remove(v)}>
                  Delete
                </button>
              </div>
            </figcaption>
          </figure>
        ))}
      </div>
      {videos.length === 0 && visibleJobs.length === 0 && <p className="muted">Your videos will appear here.</p>}
    </div>
  );
}
