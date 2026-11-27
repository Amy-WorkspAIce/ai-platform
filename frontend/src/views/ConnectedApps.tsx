import { useEffect, useState } from "react";
import type { Api, Connector } from "../api";

/**
 * Settings: apps whose tools Amy can use in chat. The hub adds the suite's apps (Docs, Sheets,
 * Slides...) by itself; any other MCP server can be added by URL.
 */
export function ConnectedApps({ api }: { api: Api }) {
  const [list, setList] = useState<Connector[] | null>(null);
  const [error, setError] = useState("");
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState("");

  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    api.connectors().then(setList).catch(fail);
  }, [api]);

  function replace(c: Connector) {
    setList((prev) => (prev ?? []).map((x) => (x.id === c.id ? c : x)));
  }

  async function run(id: string, action: () => Promise<void>) {
    setBusy(id);
    setError("");
    try {
      await action();
    } catch (e) {
      fail(e);
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="card setting">
      <div className="setting-head">
        <h2>Connected apps</h2>
        {!adding && (
          <button className="secondary" onClick={() => setAdding(true)}>
            Add
          </button>
        )}
      </div>
      <p className="muted small">
        Amy can read and change your things in these apps when you ask in chat ("add a summary to my Q3 plan"). Apps you
        install from the hub connect by themselves. Anything that serves MCP over HTTP can be added by URL.
      </p>
      {list === null ? (
        <p className="muted small">Loading…</p>
      ) : list.length === 0 && !adding ? (
        <p className="muted small">No apps connected yet.</p>
      ) : (
        <ul className="connectors">
          {list.map((c) => (
            <li key={c.id} className="connector">
              <div className="connector-head">
                <strong>{c.name}</strong>
                {c.source === "hub" && <span className="tag">From the hub</span>}
                {!c.enabled ? (
                  <span className="tag">Off</span>
                ) : c.error ? (
                  <span className="tag warn" title={c.error}>
                    Not reachable
                  </span>
                ) : (
                  <span className="tag tools">{c.tools?.length ?? 0} tools</span>
                )}
                <span className="connector-actions">
                  <button className="link small" disabled={!!busy} onClick={() => run(c.id, async () => replace(await api.testConnector(c.id)))}>
                    {busy === c.id ? "Checking…" : "Check"}
                  </button>
                  <button
                    className="link small"
                    disabled={!!busy}
                    onClick={() => run(c.id, async () => replace(await api.putConnector(c.id, { enabled: !c.enabled })))}
                  >
                    {c.enabled ? "Turn off" : "Turn on"}
                  </button>
                  <button
                    className="link small danger"
                    disabled={!!busy}
                    onClick={() =>
                      confirm(
                        c.source === "hub"
                          ? `Remove ${c.name}? The hub adds it back when it next updates the app; turning it off lasts.`
                          : `Remove ${c.name}?`,
                      ) &&
                      run(c.id, async () => {
                        await api.deleteConnector(c.id);
                        setList((prev) => (prev ?? []).filter((x) => x.id !== c.id));
                      })
                    }
                  >
                    Remove
                  </button>
                </span>
              </div>
              <code className="muted small connector-url">{c.url}</code>
              {c.enabled && c.error && <p className="error small">{c.error}</p>}
              {c.enabled && !!c.tools?.length && (
                <details className="small">
                  <summary className="muted">What Amy can do here</summary>
                  <ul className="connector-tools">
                    {c.tools.map((t) => (
                      <li key={t.name}>
                        <code>{t.name}</code> {t.description}
                        {t.changes && <span className="muted"> (asks you first)</span>}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
      {adding && (
        <AddConnector
          api={api}
          taken={(list ?? []).map((c) => c.id)}
          onAdded={(c) => {
            setList((prev) => [...(prev ?? []), c]);
            setAdding(false);
          }}
          onCancel={() => setAdding(false)}
        />
      )}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}

function AddConnector({
  api,
  taken,
  onAdded,
  onCancel,
}: {
  api: Api;
  taken: string[];
  onAdded: (c: Connector) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const id = slug(name || hostOf(url), taken);

  async function add() {
    setBusy(true);
    setError("");
    try {
      onAdded(await api.putConnector(id, { name: name.trim() || undefined, url: url.trim(), api_key: key.trim() || undefined }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="connector-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (url.trim() && id) void add();
      }}
    >
      <input placeholder="Name (e.g. Wiki)" value={name} onChange={(e) => setName(e.target.value)} />
      <input placeholder="MCP URL, e.g. http://192.168.1.20:8500/mcp" value={url} onChange={(e) => setUrl(e.target.value)} autoFocus />
      <input type="password" autoComplete="off" placeholder="API key (if it needs one)" value={key} onChange={(e) => setKey(e.target.value)} />
      <div className="row-actions">
        <button type="button" className="link" onClick={onCancel}>
          Cancel
        </button>
        <button className="primary" disabled={busy || !/^https?:\/\/./.test(url.trim()) || !id}>
          {busy ? "Connecting…" : "Connect"}
        </button>
      </div>
      {error && <p className="error small">{error}</p>}
    </form>
  );
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
}

/** A free connector id from a name: lowercase, dashes, unique. */
function slug(text: string, taken: string[]): string {
  const base = text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 24) || "app";
  let id = base;
  for (let n = 2; taken.includes(id); n++) id = `${base}-${n}`;
  return id;
}
