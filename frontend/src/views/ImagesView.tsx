import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Api, getPref, setPref, type GalleryImage, type ModelInfo } from "../api";

type Props = {
  api: Api;
  models: ModelInfo[] | null;
  onBrowseModels: () => void;
};

// Aspect presets, scaled to the model's native resolution.
const SHAPES = [
  { id: "square", label: "Square", w: 1, h: 1 },
  { id: "portrait", label: "Portrait", w: 3, h: 4 },
  { id: "landscape", label: "Landscape", w: 4, h: 3 },
  { id: "wide", label: "Wide", w: 16, h: 9 },
];

function sizeFor(shape: (typeof SHAPES)[number], base: number): string {
  // Keep roughly the model's native pixel count; round to multiples of 64.
  const scale = Math.sqrt((base * base) / (shape.w * shape.h));
  const r = (x: number) => Math.max(256, Math.round((x * scale) / 64) * 64);
  return `${r(shape.w)}x${r(shape.h)}`;
}

export function ImagesView({ api, models, onBrowseModels }: Props) {
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [shape, setShape] = useState(SHAPES[0].id);
  const [steps, setSteps] = useState("");
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [gallery, setGallery] = useState<GalleryImage[]>([]);
  const [selected, setSelected] = useState<GalleryImage | null>(null);

  const info = models?.find((m) => m.id === model);
  const galleryIds = useMemo(() => gallery.map((g) => g.id), [gallery]);
  const urls = useImageUrls(api, galleryIds);

  useEffect(() => {
    if (!models?.length || models.some((m) => m.id === model)) return;
    const last = getPref("imageModel");
    setModel(models.some((m) => m.id === last) ? last : models[0].id);
  }, [models, model]);

  const refresh = useCallback(async () => {
    try {
      setGallery(await api.images());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!busy) return;
    const start = Date.now();
    const t = setInterval(() => setElapsed(Math.round((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(t);
  }, [busy]);

  async function generate() {
    if (!prompt.trim() || !model || busy) return;
    setBusy(true);
    setElapsed(0);
    setError("");
    try {
      const shapeDef = SHAPES.find((s) => s.id === shape)!;
      await api.generateImage({
        model,
        prompt: prompt.trim(),
        size: sizeFor(shapeDef, info?.default_size ?? 1024),
        negative_prompt: info?.negative_prompt && negative.trim() ? negative.trim() : undefined,
        steps: steps ? Number(steps) : undefined,
        seed: seed ? Number(seed) : undefined,
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(img: GalleryImage) {
    if (!confirm("Delete this image?")) return;
    try {
      await api.deleteImage(img.id);
      setSelected(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function reuse(img: GalleryImage) {
    setPrompt(img.prompt);
    setNegative(img.params.negative_prompt ?? "");
    setSeed(String(img.params.seed));
    setSteps(String(img.params.steps));
    if (models?.some((m) => m.id === img.model)) setModel(img.model);
    setSelected(null);
  }

  if (models !== null && models.length === 0) {
    return (
      <div className="page">
        <h1>Images</h1>
        <div className="welcome-card left">
          <p>
            Create images from a text description. Install an image model first. Amy will
            set up the image engine for you.
          </p>
          <button className="primary" onClick={onBrowseModels}>
            Choose an image model
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Images</h1>
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
          placeholder="Describe the image… e.g. a cozy reading nook in a treehouse, golden hour, watercolor"
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void generate();
          }}
        />
        {info?.negative_prompt && (
          <input
            value={negative}
            placeholder="Avoid (optional): blurry, text, extra fingers"
            onChange={(e) => setNegative(e.target.value)}
          />
        )}
        <div className="image-controls">
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setPref("imageModel", e.target.value);
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
              <button
                key={s.id}
                type="button"
                className={shape === s.id ? "active" : ""}
                onClick={() => setShape(s.id)}
              >
                {s.label}
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
          <button type="submit" className="primary generate" disabled={!prompt.trim() || busy}>
            {busy ? `Generating… ${elapsed}s` : "Generate"}
          </button>
        </div>
        {busy && elapsed > 5 && (
          <p className="muted small">
            The first image after a while takes longer: the model has to load into memory.
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </form>

      <div className="gallery">
        {gallery.map((img) => (
          <button key={img.id} className="thumb" onClick={() => setSelected(img)} title={img.prompt}>
            {urls[img.id] ? <img src={urls[img.id]} alt={img.prompt} loading="lazy" /> : <span className="muted small">…</span>}
          </button>
        ))}
      </div>
      {gallery.length === 0 && <p className="muted">Your images will appear here.</p>}

      {selected && (
        <div className="lightbox" onClick={() => setSelected(null)}>
          <div className="lightbox-body" onClick={(e) => e.stopPropagation()}>
            {urls[selected.id] && <img src={urls[selected.id]} alt={selected.prompt} />}
            <div className="lightbox-info">
              <p>{selected.prompt}</p>
              <p className="muted small">
                {selected.model.replace("image/", "")} · {selected.params.width}×{selected.params.height} ·{" "}
                {selected.params.steps} steps · seed {selected.params.seed}
                {selected.params.seconds ? ` · ${selected.params.seconds}s` : ""}
              </p>
              <div className="row-actions start">
                <a className="button secondary" href={urls[selected.id]} download={`amy-${selected.id}.png`}>
                  Download
                </a>
                <button className="secondary" onClick={() => reuse(selected)}>
                  Reuse settings
                </button>
                <button className="link danger" onClick={() => remove(selected)}>
                  Delete
                </button>
                <button className="link" onClick={() => setSelected(null)}>
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Images need the API key, so they're fetched as blobs and shown via object URLs. */
function useImageUrls(api: Api, ids: string[]): Record<string, string> {
  const [urls, setUrls] = useState<Record<string, string>>({});
  const requested = useRef(new Set<string>());
  const live = useRef(urls);
  live.current = urls;

  useEffect(() => {
    for (const id of ids) {
      if (requested.current.has(id)) continue;
      requested.current.add(id);
      api
        .imageUrl(id)
        .then((url) => setUrls((u) => ({ ...u, [id]: url })))
        .catch(() => requested.current.delete(id));
    }
  }, [api, ids]);

  // Release blob memory when leaving the page.
  useEffect(() => () => Object.values(live.current).forEach((u) => URL.revokeObjectURL(u)), []);

  return urls;
}
