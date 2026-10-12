import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Api,
  setPref,
  type Capability,
  type Catalog,
  type Fit,
  type Hardware,
  type InstalledModel,
  type ModelRow,
  type PullState,
  type SortKey,
} from "../api";
import { ago, downloads, filterModels, loadModelTable, type FitFilter } from "../modelTable";
import { Icon } from "../icons";

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

/** Sortable columns. The first click uses the natural order; the next click reverses it. */
type Column = "perfect" | "name" | "size" | "fit" | "downloads";
type Sort = { column: Column; reverse: boolean };

const COLUMN_SORT: Record<Column, { key: SortKey; label: string; order: [string, string] }> = {
  perfect: { key: "perfect", label: "Perfect match", order: ["best picks first", "best picks last"] },
  name: { key: "name", label: "Name", order: ["A to Z", "Z to A"] },
  size: { key: "size_desc", label: "Size", order: ["largest first", "smallest first"] },
  fit: { key: "fit", label: "Fit", order: ["most headroom first", "least headroom first"] },
  downloads: { key: "popular", label: "Downloads", order: ["most first", "fewest first"] },
};

const FIT_FILTER_LABEL: Record<FitFilter, string> = {
  any: "Any fit",
  runs: "Runs on this machine",
  gpu: "Fits on GPU",
  gpu_partial: "Fits on GPU or partially",
};

function sortParams(sort: Sort): { sort: SortKey; reverse: boolean } {
  // Size reverses to its own ascending order so models with unknown size stay last.
  if (sort.column === "size") return { sort: sort.reverse ? "size_asc" : "size_desc", reverse: false };
  return { sort: COLUMN_SORT[sort.column].key, reverse: sort.reverse };
}

const SOURCE_LABEL = { curated: "Recommended", ollama: "Ollama", huggingface: "Hugging Face" } as const;
const PAGE = 50;

/** What's needed to install or use a model, from a table row or a recommendation. */
type Target = { tag: string; engine: string; use_cases: string[] };

type Actions = {
  install: (t: Target) => void;
  remove: (tag: string) => void;
  use: (t: Target) => void;
  pulls: Record<string, PullState>;
  onOpenSettings: () => void;
};

export function ModelsView({ api, onModelsChanged, onUseModel, onOpenSettings }: Props) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});
  const [error, setError] = useState("");
  const [tableVersion, setTableVersion] = useState(0);
  const [tab, setTab] = useState<"discover" | "installed">("discover");
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

  // After installs/deletes, re-fetch the table so "installed" states are current.
  const changed = useCallback(async () => {
    await refresh();
    await loadModelTable(api, true).catch(() => undefined);
    setTableVersion((v) => v + 1);
    onModelsChanged();
  }, [api, refresh, onModelsChanged]);

  const pollPulls = useCallback(async () => {
    try {
      const list = await api.pulls();
      setPulls(Object.fromEntries(list.map((p) => [p.model, p])));
      const newlyDone = list.filter((p) => p.done && !seenDone.current.has(p.model));
      newlyDone.forEach((p) => seenDone.current.add(p.model));
      if (newlyDone.some((p) => !p.error)) await changed();
    } catch {
      // transient; next poll retries
    }
  }, [api, changed]);

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

  async function install(t: Target) {
    setError("");
    try {
      seenDone.current.delete(t.tag);
      const state = await api.pull(t.tag, t.engine === "ollama" ? undefined : t.engine);
      setPulls((p) => ({ ...p, [t.tag]: state }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove(tag: string) {
    if (!confirm(`Delete ${tag}? You can download it again later.`)) return;
    try {
      await api.deleteModel(tag);
      await changed();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function use(t: Target) {
    if (t.engine === "image") {
      const video = t.use_cases.includes("video");
      setPref(video ? "videoModel" : "imageModel", `image/${t.tag}`);
      onUseModel(video ? "video" : "image");
    } else if (t.engine === "audio") {
      onUseModel(t.use_cases.includes("speech") ? "speech" : "transcription");
    } else {
      setPref("model", `ollama/${t.tag}`);
      onUseModel("chat");
    }
  }

  if (!catalog) {
    return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  }

  const byTag = new Map(catalog.families.flatMap((f) => f.variants.map((v) => [v.tag, { family: f, variant: v }] as const)));
  const actions: Actions = { install, remove, use, pulls, onOpenSettings };
  const catalogTags = new Set(byTag.keys());
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

      <div className="segmented library-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "discover"} className={tab === "discover" ? "active" : ""} onClick={() => setTab("discover")}>
          Discover
        </button>
        <button role="tab" aria-selected={tab === "installed"} className={tab === "installed" ? "active" : ""} onClick={() => setTab("installed")}>
          Installed
        </button>
      </div>

      {tab === "installed" ? (
        <InstalledModels api={api} useCases={catalog.use_cases} extra={extraInstalled} actions={actions} version={tableVersion} />
      ) : (
      <>
      <section>
        <h2>Recommended for your hardware</h2>
        <div className="rec-grid">
          {catalog.use_cases
            .filter((u) => catalog.recommendations[u.id]?.length)
            .map((u) => {
              const [top, ...alts] = catalog.recommendations[u.id];
              const { family, variant } = byTag.get(top)!;
              const target = { tag: variant.tag, engine: family.engine, use_cases: variant.use_cases };
              return (
                <div key={u.id} className="card rec">
                  <div className="rec-head">
                    <strong>{u.name}</strong>
                    <span className="muted small">{u.description}</span>
                  </div>
                  <div className="rec-pick">
                    <div>
                      <div>
                        {family.name} <span className="muted">{variant.params}</span>
                      </div>
                      <div className="muted small">{variant.size_gb} GB download</div>
                    </div>
                    <Action row={{ ...variant, ...target }} {...actions} />
                  </div>
                  {alts.length > 0 && (
                    <div className="muted small">
                      Also good: {alts.map((t) => `${byTag.get(t)!.family.name} ${byTag.get(t)!.variant.params}`).join(", ")}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </section>

      <AllModels api={api} useCases={catalog.use_cases} actions={actions} version={tableVersion} />

      <section>
        <h2>Install by name</h2>
        <CustomPull onInstall={(tag) => install({ tag, engine: "ollama", use_cases: ["chat"] })} />
      </section>
      </>
      )}
    </div>
  );
}

function AllModels({
  api,
  useCases,
  actions,
  version,
}: {
  api: Api;
  useCases: Catalog["use_cases"];
  actions: Actions;
  version: number;
}) {
  const [rows, setRows] = useState<ModelRow[] | null>(null);
  const [extra, setExtra] = useState<ModelRow[]>([]);
  const [category, setCategory] = useState("all");
  const [sort, setSort] = useState<Sort>({ column: "perfect", reverse: false });
  const [query, setQuery] = useState("");
  const [fit, setFit] = useState<FitFilter>("any");
  const [shown, setShown] = useState(PAGE);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    loadModelTable(api)
      .then((t) => !cancelled && setRows(t.models))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [api, version]);

  useEffect(() => setShown(PAGE), [category, sort, query, fit]);

  const all = useMemo(() => {
    if (!rows) return [];
    const tags = new Set(rows.map((r) => r.tag));
    return [...rows, ...extra.filter((r) => !tags.has(r.tag))];
  }, [rows, extra]);
  const filtered = useMemo(() => filterModels(all, { category, query, fit, ...sortParams(sort) }), [all, category, query, sort, fit]);
  const sortBy = (column: Column) =>
    setSort((s) => (s.column === column ? { column, reverse: !s.reverse } : { column, reverse: false }));

  async function searchHub() {
    const q = query.trim();
    setSearching(true);
    setError("");
    try {
      const r = await api.searchModels(q);
      setExtra((prev) => {
        const seen = new Set(prev.map((m) => m.tag));
        return [...prev, ...r.models.filter((m) => !seen.has(m.tag))];
      });
      setSearched(q);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  const q = query.trim();
  const sortInfo = COLUMN_SORT[sort.column];

  return (
    <section>
      <h2>All models</h2>
      <p className="muted small">The whole Ollama library plus image, video and speech models from Hugging Face, rated for your hardware.</p>
      <div className="browse-controls">
        <input className="browse-search" type="search" placeholder="Search models…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select value={fit} onChange={(e) => setFit(e.target.value as FitFilter)} aria-label="Filter by fit">
          {(Object.keys(FIT_FILTER_LABEL) as FitFilter[]).map((f) => (
            <option key={f} value={f}>
              {FIT_FILTER_LABEL[f]}
            </option>
          ))}
        </select>
      </div>
      <p className="muted small sort-hint">
        {sort.column === "perfect" ? (
          <>
            Sorted by <strong>Perfect match</strong>: the most capable models that still run well on your hardware
            {sort.reverse ? ", reversed" : ""}. Click a column title to sort by it.
          </>
        ) : (
          <>
            Sorted by <strong>{sortInfo.label}</strong>, {sortInfo.order[sort.reverse ? 1 : 0]}.{" "}
            <button className="link small" onClick={() => setSort({ column: "perfect", reverse: false })}>
              Back to Perfect match
            </button>
          </>
        )}
      </p>
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
      {!rows ? (
        <p className="muted">Loading models…</p>
      ) : (
        <>
          <p className="muted small">{filtered.length.toLocaleString()} models</p>
          <div className="model-table" role="table">
            <div className="model-row head" role="row">
              <SortHeader column="name" sort={sort} onSort={sortBy} label="Model" />
              <SortHeader column="size" sort={sort} onSort={sortBy} label="Size" />
              <SortHeader column="fit" sort={sort} onSort={sortBy} label="Fit" />
              <SortHeader column="downloads" sort={sort} onSort={sortBy} label="Downloads" />
              <SortHeader column="perfect" sort={sort} onSort={sortBy} label="Match" />
            </div>
            {filtered.slice(0, shown).map((r) => (
              <Row key={r.tag} row={r} useCases={useCases} {...actions} />
            ))}
          </div>
          {filtered.length > shown && (
            <button className="secondary load-more" onClick={() => setShown((n) => n + PAGE)}>
              Show more ({(filtered.length - shown).toLocaleString()} more)
            </button>
          )}
          {q.length >= 2 && searched !== q && (
            <div className="hub-search">
              <span className="muted small">Not finding it?</span>
              <button className="secondary" disabled={searching} onClick={searchHub}>
                {searching ? "Searching…" : `Search Hugging Face for "${q}"`}
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function Row({ row: r, useCases, ...actions }: { row: ModelRow; useCases: Catalog["use_cases"] } & Actions) {
  const names = Object.fromEntries(useCases.map((u) => [u.id, u.id === "chat" ? "General" : u.name]));
  const source = r.source ?? "curated";
  const dl = downloads(r);
  return (
    <div className="model-row" role="row">
      <div className="model-cell-main">
        <div className="model-name">
          <strong>{r.name}</strong>
          {r.params && <span className="model-params">{r.params}</span>}
          <span className={`tag source-${source}`}>{SOURCE_LABEL[source]}</span>
          {r.tools && (
            <span className="tag tools" title="Can use web search and agent mode">
              Tools
            </span>
          )}
          {r.url && (
            <a className="small" href={r.url} target="_blank" rel="noreferrer" title="Open model page">
              <Icon name="external" size={14} label="Open model page" />
            </a>
          )}
        </div>
        <div className="model-meta muted small" title={r.description}>
          {r.use_cases.map((u) => names[u] ?? u).join(" · ")}
          {r.publisher ? ` · ${r.publisher}` : ""}
          {r.license ? ` · ${r.license}` : ""}
          {r.updated ? ` · updated ${ago(r.updated)}` : ""}
          {r.description ? ` · ${r.description}` : ""}
        </div>
        {(r.unsupported || r.engine_note) && <div className="engine-note small">{r.unsupported || r.engine_note}</div>}
      </div>
      <span className="muted small cell-size" title={r.size_estimated ? "Estimated; exact size is known at install" : undefined}>
        {r.size_gb != null ? `${r.size_estimated ? "~" : ""}${r.size_gb} GB` : "?"}
      </span>
      <span className="cell-fit">
        <span className={`fit fit-${r.fit}`} title={r.needed_gb ? `Needs about ${r.needed_gb} GB of memory` : undefined}>
          {FIT_LABEL[r.fit]}
        </span>
      </span>
      <span
        className="muted small cell-downloads"
        title={r.downloads_scope === "family" ? `${dl.toLocaleString()} across all sizes of ${r.name} (per-size counts aren't published)` : dl ? `${dl.toLocaleString()} downloads` : undefined}
      >
        {dl ? compact(dl) : "–"}
        {dl > 0 && r.downloads_scope === "family" && <span className="dl-scope"> all sizes</span>}
      </span>
      {r.unsupported ? <span className="row-actions muted small">Not supported yet</span> : <Action row={r} {...actions} />}
    </div>
  );
}

type ActionRow = Target & {
  installed: boolean;
  requires_token?: boolean;
  fit: Fit;
  disk_ok: boolean;
};

function Action({ row: r, install, remove, use, pulls, onOpenSettings }: { row: ActionRow } & Actions) {
  const pull = pulls[r.tag];
  if (pull && !pull.done) {
    return (
      <span className="row-actions">
        <Progress value={pull.total ? pull.completed / pull.total : 0} small />
      </span>
    );
  }
  if (r.installed) {
    return (
      <span className="row-actions">
        {/* Media models are used automatically when you ask in chat (Settings can pin one). */}
        {r.engine === "ollama" && !r.use_cases.includes("embedding") && (
          <button className="secondary" onClick={() => use(r)}>
            Chat
          </button>
        )}
        <button className="link danger" onClick={() => remove(r.tag)}>
          Delete
        </button>
      </span>
    );
  }
  if (r.requires_token) {
    return (
      <span className="row-actions">
        <button className="secondary token-needed" onClick={onOpenSettings} title="This model is gated. Add a Hugging Face token in Settings.">
          Requires Hugging Face token
        </button>
      </span>
    );
  }
  const risky = r.fit === "too_large" || !r.disk_ok;
  return (
    <span className="row-actions">
      <button
        className={risky ? "secondary risky" : "secondary"}
        title={!r.disk_ok ? "Not enough free disk space" : undefined}
        onClick={() => {
          if (risky && !confirm(`${r.tag} probably won't run well on this machine${r.disk_ok ? "" : " and may not fit on disk"}. Download anyway?`)) return;
          install(r);
        }}
      >
        Install
      </button>
    </span>
  );
}

function SortHeader({ column, label, sort, onSort }: { column: Column; label: string; sort: Sort; onSort: (c: Column) => void }) {
  const active = sort.column === column;
  return (
    <button
      role="columnheader"
      className={`sort-header ${active ? "active" : ""}`}
      aria-sort={active ? (sort.reverse ? "descending" : "ascending") : "none"}
      title={`Sort by ${COLUMN_SORT[column].label.toLowerCase()} (${COLUMN_SORT[column].order[active && !sort.reverse ? 1 : 0]})`}
      onClick={() => onSort(column)}
    >
      {label}
      <span className="sort-arrow" aria-hidden>
        {active ? (sort.reverse ? "\u25B4" : "\u25BE") : ""}
      </span>
    </button>
  );
}

/** Everything installed, with what it's used for, its size and fit, and actions. */
function InstalledModels({
  api,
  useCases,
  extra,
  actions,
  version,
}: {
  api: Api;
  useCases: Catalog["use_cases"];
  extra: InstalledModel[];
  actions: Actions;
  version: number;
}) {
  const [rows, setRows] = useState<ModelRow[] | null>(null);
  const [defaults, setDefaults] = useState<Record<string, string>>({});
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    loadModelTable(api)
      .then((t) => !cancelled && setRows(t.models.filter((r) => r.installed)))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    api
      .settings()
      .then((s) => {
        if (cancelled) return;
        const used: Record<string, string> = {};
        for (const [cap, choice] of Object.entries(s.models)) {
          const id = choice.pinned ?? choice.best;
          if (id) used[id] = [used[id], cap].filter(Boolean).join(", ");
        }
        setDefaults(used);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [api, version]);

  if (error) return <p className="error">{error}</p>;
  if (!rows) return <p className="muted">Loading…</p>;

  const names = Object.fromEntries(useCases.map((u) => [u.id, u.id === "chat" ? "Chat" : u.name]));
  const total = rows.reduce((n, r) => n + (r.size_gb ?? 0), 0) + extra.reduce((n, m) => n + m.size_gb, 0);
  const count = rows.length + extra.length;

  return (
    <section>
      <p className="muted small">
        {count} model{count === 1 ? "" : "s"} installed, about {Math.round(total * 10) / 10} GB on disk. Chat hands images,
        video and speech to the best installed model for each (or the one you pick in Settings).
      </p>
      {count === 0 ? (
        <p className="muted">Nothing installed yet. Find models under Discover.</p>
      ) : (
        <div className="model-table installed-table" role="table">
          <div className="model-row head" role="row">
            <span>Model</span>
            <span>Size</span>
            <span>Fit</span>
            <span>Used for</span>
            <span />
          </div>
          {rows.map((r) => {
            const used = defaults[`${r.engine}/${r.tag}`];
            return (
              <div key={r.tag} className="model-row" role="row">
                <div className="model-cell-main">
                  <div className="model-name">
                    <strong>{r.name}</strong>
                    {r.params && <span className="model-params">{r.params}</span>}
                  </div>
                  <div className="model-meta muted small">
                    <span className="mono">{r.tag}</span> · {r.use_cases.map((u) => names[u] ?? u).join(" · ")}
                  </div>
                </div>
                <span className="muted small cell-size">{r.size_gb != null ? `${r.size_gb} GB` : "?"}</span>
                <span className="cell-fit">
                  <span className={`fit fit-${r.fit}`}>{FIT_LABEL[r.fit]}</span>
                </span>
                <span className="small cell-downloads">
                  {used ? <span className="tag source-curated">Default: {used}</span> : <span className="muted">–</span>}
                </span>
                <Action row={r} {...actions} />
              </div>
            );
          })}
          {extra.map((m) => (
            <div key={m.name} className="model-row" role="row">
              <div className="model-cell-main">
                <div className="model-name">
                  <strong className="mono">{m.name}</strong>
                </div>
                <div className="model-meta muted small">{[m.parameters, m.quantization, "Installed by name"].filter(Boolean).join(" · ")}</div>
              </div>
              <span className="muted small cell-size">{m.size_gb} GB</span>
              <span className="cell-fit" />
              <span className="muted small cell-downloads">–</span>
              <span className="row-actions">
                <button className="secondary" onClick={() => actions.use({ tag: m.name, engine: "ollama", use_cases: ["chat"] })}>
                  Chat
                </button>
                <button className="link danger" onClick={() => actions.remove(m.name)}>
                  Delete
                </button>
              </span>
            </div>
          ))}
        </div>
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
          Models will run on the CPU, which is much slower. Have an NVIDIA GPU? Make sure the container runs with GPU access
          (Unraid: <code>--runtime=nvidia</code>; Docker: <code>--gpus all</code>).
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
      <input value={value} placeholder="e.g. llama3.2:3b or hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M" onChange={(e) => setValue(e.target.value)} />
      <button type="submit" className="secondary" disabled={!value.trim()}>
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
