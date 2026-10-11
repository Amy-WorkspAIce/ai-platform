import { useCallback, useEffect, useRef, useState } from "react";
import {
  Api,
  setPref,
  type BrowseResult,
  type Capability,
  type Catalog,
  type Family,
  type Fit,
  type Hardware,
  type InstalledModel,
  type PullState,
  type SortKey,
  type Variant,
} from "../api";

type Props = {
  api: Api;
  onModelsChanged: () => void;
  onUseModel: (capability: Capability) => void;
  onOpenSettings: () => void;
};

const FIT_LABEL: Record<Fit, string> = {
  gpu: "Runs on GPU",
  light: "Runs anywhere",
  partial: "Partial GPU · slower",
  cpu: "Runs on CPU · slow",
  too_large: "Too large",
  unknown: "Unknown fit",
};

const SORTS: { id: SortKey; label: string; hint: string }[] = [
  { id: "perfect", label: "Perfect match", hint: "The most capable models that still run well on your hardware" },
  { id: "fit", label: "Best fit", hint: "Most headroom first" },
  { id: "popular", label: "Most popular", hint: "Downloads / pulls" },
  { id: "size_desc", label: "Largest first", hint: "" },
  { id: "size_asc", label: "Smallest first", hint: "" },
  { id: "name", label: "Name", hint: "" },
];

const SOURCE_LABEL = { curated: "Recommended", ollama: "Ollama", huggingface: "Hugging Face" } as const;

type Actions = {
  install: (v: Variant, f: Family) => void;
  remove: (tag: string) => void;
  use: (v: Variant, f: Family) => void;
  pulls: Record<string, PullState>;
  hfTokenSet: boolean;
  onOpenSettings: () => void;
};

export function ModelsView({ api, onModelsChanged, onUseModel, onOpenSettings }: Props) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});
  const [error, setError] = useState("");
  const [browseKey, setBrowseKey] = useState(0);
  const seenDone = useRef(new Set<string>());

  const refresh = useCallback(async () => {
    try {
      const cat = await api.catalog();
      setCatalog(cat);
      setInstalled(cat.ollama_online ? await api.installed() : []);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  const pollPulls = useCallback(async () => {
    try {
      const list = await api.pulls();
      setPulls(Object.fromEntries(list.map((p) => [p.model, p])));
      const newlyDone = list.filter((p) => p.done && !seenDone.current.has(p.model));
      newlyDone.forEach((p) => seenDone.current.add(p.model));
      if (newlyDone.some((p) => !p.error)) {
        await refresh();
        setBrowseKey((k) => k + 1);
        onModelsChanged();
      }
    } catch {
      // transient; next poll retries
    }
  }, [api, refresh, onModelsChanged]);

  useEffect(() => {
    void refresh();
    void pollPulls();
  }, [refresh, pollPulls]);

  const active = Object.values(pulls).some((p) => !p.done);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => void pollPulls(), 1000);
    return () => clearInterval(t);
  }, [active, pollPulls]);

  async function install(v: Variant, f: Family) {
    setError("");
    try {
      seenDone.current.delete(v.tag);
      const engine = f.engine === "ollama" ? undefined : f.engine;
      const state = await api.pull(v.tag, engine);
      setPulls((p) => ({ ...p, [v.tag]: state }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove(tag: string) {
    if (!confirm(`Delete ${tag}? You can download it again later.`)) return;
    try {
      await api.deleteModel(tag);
      await refresh();
      setBrowseKey((k) => k + 1);
      onModelsChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function use(v: Variant, f: Family) {
    if (f.engine === "image") {
      const video = f.use_cases.includes("video");
      setPref(video ? "videoModel" : "imageModel", `image/${v.tag}`);
      onUseModel(video ? "video" : "image");
    } else if (f.engine === "audio") {
      onUseModel(v.use_cases.includes("speech") ? "speech" : "transcription");
    } else {
      setPref("model", `ollama/${v.tag}`);
      onUseModel("chat");
    }
  }

  if (!catalog) {
    return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  }

  const variantsByTag = new Map(
    catalog.families.flatMap((f) => f.variants.map((v) => [v.tag, { family: f, variant: v }] as const)),
  );
  const actions: Actions = { install, remove, use, pulls, hfTokenSet: catalog.hf_token_set, onOpenSettings };
  const catalogTags = new Set(variantsByTag.keys());
  const extraInstalled = installed.filter((m) => !catalogTags.has(m.name));

  return (
    <div className="page">
      <h1>Models</h1>
      <HardwareCard hw={catalog.hardware} />
      {!catalog.ollama_online && (
        <p className="error">The Ollama engine isn't reachable, so chat models can't be installed right now.</p>
      )}
      {error && <p className="error">{error}</p>}

      <ActivePulls pulls={Object.values(pulls)} />

      <section>
        <h2>Recommended for your hardware</h2>
        <div className="rec-grid">
          {catalog.use_cases
            .filter((u) => catalog.recommendations[u.id]?.length)
            .map((u) => {
              const [top, ...alts] = catalog.recommendations[u.id];
              const entry = variantsByTag.get(top)!;
              return (
                <div key={u.id} className="card rec">
                  <div className="rec-head">
                    <strong>{u.name}</strong>
                    <span className="muted small">{u.description}</span>
                  </div>
                  <div className="rec-pick">
                    <div>
                      <div>
                        {entry.family.name} <span className="muted">{entry.variant.params}</span>
                      </div>
                      <div className="muted small">{entry.variant.size_gb} GB download</div>
                    </div>
                    <VariantAction v={entry.variant} f={entry.family} {...actions} />
                  </div>
                  {alts.length > 0 && (
                    <div className="muted small">
                      Also good:{" "}
                      {alts
                        .map((t) => `${variantsByTag.get(t)!.family.name} ${variantsByTag.get(t)!.variant.params}`)
                        .join(", ")}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </section>

      <Browse key={browseKey} api={api} useCases={catalog.use_cases} actions={actions} />

      <section>
        <h2>Install by name</h2>
        <CustomPull onInstall={(tag) => install({ tag } as Variant, { engine: "ollama" } as Family)} />
        {extraInstalled.length > 0 && (
          <>
            <h3>Other installed models</h3>
            <div className="card">
              {extraInstalled.map((m) => (
                <div key={m.name} className="variant-row">
                  <span className="mono">{m.name}</span>
                  <span className="muted small">
                    {[m.parameters, m.quantization, `${m.size_gb} GB`].filter(Boolean).join(" · ")}
                  </span>
                  <span className="row-actions">
                    <button className="secondary" onClick={() => use({ tag: m.name, use_cases: ["chat"] } as unknown as Variant, { engine: "ollama", use_cases: ["chat"] } as unknown as Family)}>
                      Use
                    </button>
                    <button className="link danger" onClick={() => remove(m.name)}>
                      Delete
                    </button>
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function Browse({ api, useCases, actions }: { api: Api; useCases: Catalog["use_cases"]; actions: Actions }) {
  const [category, setCategory] = useState("all");
  const [sort, setSort] = useState<SortKey>("perfect");
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [runnable, setRunnable] = useState(false);
  const [result, setResult] = useState<BrowseResult | null>(null);
  const [families, setFamilies] = useState<Family[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const PAGE = 30;

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 400);
    return () => clearTimeout(t);
  }, [query]);

  const load = useCallback(
    async (offset: number) => {
      setLoading(true);
      setError("");
      try {
        const r = await api.browse({ category, sort, q: debounced, runnable, offset, limit: PAGE });
        setResult(r);
        setFamilies((prev) => (offset === 0 ? r.families : [...prev, ...r.families]));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [api, category, sort, debounced, runnable],
  );

  useEffect(() => {
    void load(0);
  }, [load]);

  const sortHint = SORTS.find((s) => s.id === sort)?.hint;

  return (
    <section>
      <h2>Browse all models</h2>
      <p className="muted small">
        The whole Ollama library plus image, video and speech models from Hugging Face, rated for your hardware.
      </p>
      <div className="browse-controls">
        <input
          className="browse-search"
          type="search"
          placeholder="Search models… (also searches Hugging Face GGUF models)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label="Sort">
          {SORTS.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
        <label className="auto-approve">
          <input type="checkbox" checked={runnable} onChange={(e) => setRunnable(e.target.checked)} />
          Only models that run here
        </label>
      </div>
      {sortHint && <p className="muted small sort-hint">{sortHint}</p>}
      <div className="chips">
        <button className={category === "all" ? "chip active" : "chip"} onClick={() => setCategory("all")}>
          All
        </button>
        {useCases.map((u) => (
          <button key={u.id} className={category === u.id ? "chip active" : "chip"} onClick={() => setCategory(u.id)}>
            {u.id === "chat" ? "General" : u.name}
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <p className="muted small">
          {result.total.toLocaleString()} model families{debounced ? ` matching "${debounced}"` : ""}
        </p>
      )}
      <div className="family-list">
        {families.map((f) => (
          <FamilyCard key={f.id} family={f} useCases={useCases} {...actions} />
        ))}
      </div>
      {loading && <p className="muted">Loading…</p>}
      {result && families.length < result.total && !loading && (
        <button className="secondary load-more" onClick={() => void load(families.length)}>
          Show more ({result.total - families.length} more)
        </button>
      )}
    </section>
  );
}

function HardwareCard({ hw }: { hw: Hardware }) {
  const gpu = hw.gpus.length ? hw.gpus.map((g) => `${g.name} (${g.vram_total_gb} GB)`).join(", ") : "No NVIDIA GPU detected";
  return (
    <div className="card hardware">
      <div>
        <div className="muted small">GPU</div>
        <div>
          {gpu}
          {hw.vram_override && <span className="muted small"> · set by AMY_GPU_VRAM_GB</span>}
        </div>
      </div>
      <div>
        <div className="muted small">Memory</div>
        <div>{hw.ram_total_gb ? `${hw.ram_total_gb} GB` : "Unknown"}</div>
      </div>
      <div>
        <div className="muted small">Free disk</div>
        <div>{hw.disk_free_gb != null ? `${hw.disk_free_gb} GB` : "Unknown"}</div>
      </div>
      {!hw.gpus.length && (
        <p className="muted small full">
          Models will run on the CPU, which is much slower. Have an NVIDIA GPU? Make sure the container runs with GPU
          access (Unraid: <code>--runtime=nvidia</code>; Docker: <code>--gpus all</code>).
        </p>
      )}
    </div>
  );
}

function ActivePulls({ pulls }: { pulls: PullState[] }) {
  const visible = pulls.filter((p) => !p.done || p.error);
  if (!visible.length) return null;
  return (
    <section className="card">
      <h2 className="small-heading">Downloads</h2>
      {visible.map((p) => (
        <div key={p.model} className="pull">
          <div className="pull-head">
            <span className="mono">{p.model}</span>
            <span className={p.error ? "error small" : "muted small"}>
              {p.error ?? `${p.status}${p.total ? ` · ${gb(p.completed)} / ${gb(p.total)} GB` : ""}`}
            </span>
          </div>
          {!p.error && <Progress value={p.total ? p.completed / p.total : 0} />}
        </div>
      ))}
    </section>
  );
}

function FamilyCard({ family, useCases, ...actions }: { family: Family; useCases: Catalog["use_cases"] } & Actions) {
  const names = Object.fromEntries(useCases.map((u) => [u.id, u.id === "chat" ? "General" : u.name]));
  const source = family.source ?? "curated";
  return (
    <div className="card family">
      <div className="family-head">
        <div className="family-title">
          <strong>{family.name}</strong>
          <span className="muted small"> by {family.publisher}</span>
          <span className={`tag source-${source}`}>{SOURCE_LABEL[source]}</span>
          {family.url && (
            <a className="small" href={family.url} target="_blank" rel="noreferrer" title="Open model page">
              ↗
            </a>
          )}
        </div>
        <div className="tags">
          {family.use_cases.map((u) => (
            <span key={u} className="tag">
              {names[u] ?? u}
            </span>
          ))}
          {family.tools && (
            <span className="tag tools" title="Can use web search and agent mode">
              Tools
            </span>
          )}
        </div>
      </div>
      <p className="muted small">
        {family.description}
        {family.license && <span className="license">License: {family.license}</span>}
        {!!family.popularity && <span className="license">{compact(family.popularity)} downloads</span>}
      </p>
      {family.engine_note && <p className="engine-note small">{family.engine_note}</p>}
      {family.unsupported && <p className="engine-note small">{family.unsupported}</p>}
      {family.variants.map((v) => (
        <div key={v.tag} className={`variant-row ${v.best ? "best" : ""}`}>
          <span>
            {v.params || family.name}
            {v.best && <span className="tag best-tag" title="The biggest model that runs well on your hardware">Perfect match</span>}
          </span>
          <span className="muted small" title={v.size_estimated ? "Estimated; exact size is known at install" : undefined}>
            {v.size_gb != null ? `${v.size_estimated ? "~" : ""}${v.size_gb} GB` : "?"}
          </span>
          <span className={`fit fit-${v.fit}`} title={v.needed_gb ? `Needs about ${v.needed_gb} GB of memory` : undefined}>
            {FIT_LABEL[v.fit]}
          </span>
          {family.unsupported ? <span className="row-actions muted small">Not supported yet</span> : <VariantAction v={v} f={family} {...actions} />}
        </div>
      ))}
    </div>
  );
}

function VariantAction({ v, f, install, remove, use, pulls, onOpenSettings }: { v: Variant; f: Family } & Actions) {
  const pull = pulls[v.tag];
  if (pull && !pull.done) {
    return (
      <span className="row-actions">
        <Progress value={pull.total ? pull.completed / pull.total : 0} small />
      </span>
    );
  }
  if (v.installed) {
    return (
      <span className="row-actions">
        {!v.use_cases.includes("embedding") && (
          <button className="secondary" onClick={() => use(v, f)}>
            Use
          </button>
        )}
        <button className="link danger" onClick={() => remove(v.tag)}>
          Delete
        </button>
      </span>
    );
  }
  if (v.requires_token) {
    return (
      <span className="row-actions">
        <button className="secondary token-needed" onClick={onOpenSettings} title="This model is gated. Add a Hugging Face token in Settings.">
          Requires Hugging Face token
        </button>
      </span>
    );
  }
  const risky = v.fit === "too_large" || !v.disk_ok;
  return (
    <span className="row-actions">
      <button
        className={risky ? "secondary" : "primary"}
        title={!v.disk_ok ? "Not enough free disk space" : undefined}
        onClick={() => {
          if (
            risky &&
            !confirm(`${v.tag} probably won't run well on this machine${v.disk_ok ? "" : " and may not fit on disk"}. Download anyway?`)
          )
            return;
          install(v, f);
        }}
      >
        Install
      </button>
    </span>
  );
}

function CustomPull({ onInstall }: { onInstall: (tag: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <form
      className="custom-pull"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) {
          onInstall(value.trim());
          setValue("");
        }
      }}
    >
      <input
        value={value}
        placeholder="e.g. llama3.2:3b or hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
        onChange={(e) => setValue(e.target.value)}
      />
      <button type="submit" className="primary" disabled={!value.trim()}>
        Install
      </button>
      <p className="muted small full">
        Any Ollama tag, or a GGUF repo on Hugging Face (<code>hf.co/user/repo:quant</code>).
      </p>
    </form>
  );
}

function Progress({ value, small }: { value: number; small?: boolean }) {
  return (
    <div className={`progress ${small ? "small" : ""}`}>
      <div style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
    </div>
  );
}

const gb = (bytes: number) => (bytes / 1e9).toFixed(2);

function compact(n: number): string {
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}
