import { useCallback, useEffect, useRef, useState } from "react";
import {
  Api,
  setPref,
  type Capability,
  type Catalog,
  type Family,
  type Fit,
  type Hardware,
  type InstalledModel,
  type PullState,
  type Variant,
} from "../api";

type Props = {
  api: Api;
  onModelsChanged: () => void;
  onUseModel: (capability: Capability) => void;
};

const FIT_LABEL: Record<Fit, string> = {
  gpu: "Runs on GPU",
  light: "Runs anywhere",
  partial: "Partial GPU · slower",
  cpu: "Runs on CPU · slow",
  too_large: "Too large",
  unknown: "Unknown fit",
};

export function ModelsView({ api, onModelsChanged, onUseModel }: Props) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState("");
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
        onModelsChanged();
      }
      return list.some((p) => !p.done);
    } catch {
      return false;
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

  async function install(tag: string) {
    setError("");
    try {
      seenDone.current.delete(tag);
      const state = await api.pull(tag);
      setPulls((p) => ({ ...p, [tag]: state }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove(tag: string) {
    if (!confirm(`Delete ${tag}? You can download it again later.`)) return;
    try {
      await api.deleteModel(tag);
      await refresh();
      onModelsChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function use(tag: string) {
    const entry = variantsByTag.get(tag);
    const engine = entry?.family.engine ?? "ollama";
    if (engine === "image") {
      setPref("imageModel", `image/${tag}`);
      onUseModel("image");
    } else if (engine === "audio") {
      onUseModel(entry!.variant.use_cases.includes("speech") ? "speech" : "transcription");
    } else {
      setPref("model", `ollama/${tag}`);
      onUseModel("chat");
    }
  }

  const variantsByTag = new Map(
    (catalog?.families ?? []).flatMap((f) => f.variants.map((v) => [v.tag, { family: f, variant: v }] as const)),
  );

  if (!catalog) {
    return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  }

  const families = catalog.families.filter((f) => filter === "all" || f.use_cases.includes(filter));
  const catalogTags = new Set(variantsByTag.keys());
  const extraInstalled = installed.filter((m) => !catalogTags.has(m.name));
  const actions = { install, remove, use, pulls };

  return (
    <div className="page">
      <h1>Models</h1>
      <HardwareCard hw={catalog.hardware} />
      {!catalog.ollama_online && (
        <p className="error">The Ollama engine isn't reachable, so models can't be installed right now.</p>
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
                      <div>{entry.family.name} <span className="muted">{entry.variant.params}</span></div>
                      <div className="muted small">{entry.variant.size_gb} GB download</div>
                    </div>
                    <VariantAction v={entry.variant} {...actions} />
                  </div>
                  {alts.length > 0 && (
                    <div className="muted small">
                      Also good: {alts.map((t) => variantsByTag.get(t)!.family.name + " " + variantsByTag.get(t)!.variant.params).join(", ")}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </section>

      <section>
        <h2>Browse</h2>
        <div className="chips">
          <button className={filter === "all" ? "chip active" : "chip"} onClick={() => setFilter("all")}>
            All
          </button>
          {catalog.use_cases.map((u) => (
            <button
              key={u.id}
              className={filter === u.id ? "chip active" : "chip"}
              onClick={() => setFilter(u.id)}
            >
              {u.name}
            </button>
          ))}
        </div>
        <div className="family-list">
          {families.map((f) => (
            <FamilyCard key={f.id} family={f} useCases={catalog.use_cases} {...actions} />
          ))}
        </div>
      </section>

      <section>
        <h2>Install any model</h2>
        <CustomPull onInstall={install} />
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
                    <button className="secondary" onClick={() => use(m.name)}>Use</button>
                    <button className="link danger" onClick={() => remove(m.name)}>Delete</button>
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

function HardwareCard({ hw }: { hw: Hardware }) {
  const gpu = hw.gpus.length
    ? hw.gpus.map((g) => `${g.name} (${g.vram_total_gb} GB)`).join(", ")
    : "No NVIDIA GPU detected";
  return (
    <div className="card hardware">
      <div>
        <div className="muted small">GPU</div>
        <div>{gpu}{hw.vram_override && <span className="muted small"> · set by AMY_GPU_VRAM_GB</span>}</div>
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
          Models will run on the CPU, which is much slower. Have an NVIDIA GPU? Make sure the container
          runs with GPU access (Unraid: <code>--runtime=nvidia</code>; Docker: <code>--gpus all</code>).
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

function FamilyCard({
  family,
  useCases,
  ...actions
}: { family: Family; useCases: Catalog["use_cases"] } & ActionProps) {
  const names = Object.fromEntries(useCases.map((u) => [u.id, u.name]));
  return (
    <div className="card family">
      <div className="family-head">
        <div>
          <strong>{family.name}</strong> <span className="muted small">by {family.publisher}</span>
        </div>
        <div className="tags">
          {family.use_cases.map((u) => (
            <span key={u} className="tag">{names[u] ?? u}</span>
          ))}
          {!family.engine_available && <span className="tag soon">Coming soon</span>}
        </div>
      </div>
      <p className="muted small">{family.description} <span className="license">License: {family.license}</span></p>
      {family.engine_note && <p className="engine-note small">{family.engine_note}</p>}
      {family.variants.map((v) => (
        <div key={v.tag} className="variant-row">
          <span>{v.params}</span>
          <span className="muted small">{v.size_gb} GB</span>
          {family.engine_available ? (
            <>
              <span className={`fit fit-${v.fit}`} title={`Needs about ${v.needed_gb} GB of memory`}>
                {FIT_LABEL[v.fit]}
              </span>
              <VariantAction v={v} {...actions} />
            </>
          ) : (
            <span className="muted small row-actions">Engine support planned</span>
          )}
        </div>
      ))}
    </div>
  );
}

type ActionProps = {
  install: (tag: string) => void;
  remove: (tag: string) => void;
  use: (tag: string) => void;
  pulls: Record<string, PullState>;
};

function VariantAction({ v, install, remove, use, pulls }: { v: Variant } & ActionProps) {
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
          <button className="secondary" onClick={() => use(v.tag)}>Use</button>
        )}
        <button className="link danger" onClick={() => remove(v.tag)}>Delete</button>
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
          if (risky && !confirm(`${v.tag} probably won't run well on this machine${v.disk_ok ? "" : " and may not fit on disk"}. Download anyway?`)) return;
          install(v.tag);
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
        Any model from <a href="https://ollama.com/library" target="_blank" rel="noreferrer">ollama.com/library</a> or
        a GGUF repo on Hugging Face (<code>hf.co/user/repo:quant</code>).
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
