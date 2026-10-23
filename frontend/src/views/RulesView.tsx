import { useEffect, useState } from "react";
import type { Api, Folder, Rule, RuleApplies, RuleListing, WorkspaceInfo } from "../api";

const APPLIES: Record<RuleApplies, string> = {
  always: "Every chat and task",
  chat: "Chats only",
  agent: "Workspace tasks only",
};

const EXAMPLES = [
  "Before answering anything technical, ask two other models with ask_models and weigh their answers.",
  "When creating pull requests, never force-push.",
  "Answer in British English and keep replies under 200 words unless I ask for more.",
];

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function scopeLabel(scope: string, folders: Folder[]) {
  if (scope === "global") return "Everywhere";
  const [kind, name] = [scope.slice(0, scope.indexOf(":")), scope.slice(scope.indexOf(":") + 1)];
  if (kind === "folder") return `Folder: ${folders.find((f) => f.id === name)?.name ?? "deleted folder"}`;
  return `Workspace: ${name}`;
}

/**
 * Standing instructions for Amy (like Cursor's rules): everywhere, or for one sidebar folder
 * or agent workspace. Folders and workspaces inherit the rules that apply everywhere and can
 * switch any of them off for themselves.
 */
export function RulesView({ api, folders, initialScope }: { api: Api; folders: Folder[]; initialScope: string }) {
  const [scope, setScope] = useState(initialScope);
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [listing, setListing] = useState<RuleListing | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .workspaces()
      .then(setWorkspaces)
      .catch(() => setWorkspaces([]));
  }, [api]);

  const load = () => {
    setError("");
    api
      .rules(scope)
      .then(setListing)
      .catch((e) => {
        setListing(null);
        setError(errorText(e));
      });
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(load, [api, scope]);

  async function act(fn: () => Promise<unknown>) {
    setError("");
    try {
      await fn();
      load();
    } catch (e) {
      setError(errorText(e));
    }
  }

  const scopes = ["global", ...folders.map((f) => `folder:${f.id}`), ...workspaces.map((w) => `workspace:${w.name}`)];
  if (!scopes.includes(scope)) scopes.push(scope);
  const here = scope === "global" ? "everywhere" : scope.startsWith("folder:") ? "in this folder" : "in this workspace";

  return (
    <div className="page settings rules-page">
      <h1>Rules</h1>
      <p className="muted small">
        Instructions Amy follows without being reminded. Rules for <strong>everywhere</strong> also apply in every folder
        and workspace, unless you switch them off there. Folders and workspaces can add rules of their own.
      </p>
      <label className="model-default rules-scope">
        <span>
          <strong>Showing rules for</strong>
        </span>
        <select value={scope} onChange={(e) => setScope(e.target.value)}>
          {scopes.map((s) => (
            <option key={s} value={s}>
              {scopeLabel(s, folders)}
            </option>
          ))}
        </select>
      </label>
      {error && <p className="error small">{error}</p>}
      {!listing ? (
        !error && <p className="muted">Loading…</p>
      ) : (
        <>
          <section className="card setting">
            <h2>Rules {here}</h2>
            {listing.rules.length === 0 && <p className="muted small">No rules {here} yet.</p>}
            <div className="rule-list">
              {listing.rules.map((r) => (
                <RuleRow key={r.id} rule={r} api={api} onChanged={load} onError={setError} />
              ))}
            </div>
            <NewRule onAdd={(rule) => act(() => api.addRule({ ...rule, scope }))} />
          </section>

          {scope !== "global" && (
            <section className="card setting">
              <h2>From everywhere</h2>
              {listing.inherited.length === 0 ? (
                <p className="muted small">There are no rules for everywhere.</p>
              ) : (
                <>
                  <p className="muted small">Untick a rule to switch it off {here} only.</p>
                  <div className="rule-list">
                    {listing.inherited.map((r) => (
                      <label key={r.id} className={`rule inherited ${!r.enabled ? "off" : ""}`}>
                        <input
                          type="checkbox"
                          checked={!r.excluded}
                          disabled={!r.enabled}
                          onChange={(e) => void act(() => api.setRuleExcluded(r.id, scope, !e.target.checked))}
                        />
                        <span>
                          <strong>{r.name}</strong>
                          <span className="muted small"> · {r.enabled ? APPLIES[r.applies] : "switched off everywhere"}</span>
                          <span className="rule-text">{r.text}</span>
                        </span>
                      </label>
                    ))}
                  </div>
                </>
              )}
            </section>
          )}

          {scope.startsWith("workspace:") && (
            <section className="card setting">
              <h2>From the workspace's files</h2>
              {listing.project.length === 0 ? (
                <p className="muted small">
                  Rule files in the workspace (<code>AGENTS.md</code>, <code>CLAUDE.md</code>, <code>.cursorrules</code>,{" "}
                  <code>.cursor/rules/</code>) are followed automatically. This one has none.
                </p>
              ) : (
                <>
                  <p className="muted small">Followed automatically in agent tasks here. Edit the files to change them.</p>
                  {listing.project.map((f) => (
                    <details key={f.name} className="rule project">
                      <summary>
                        <code>{f.name}</code>
                      </summary>
                      <pre className="rule-text">{f.text}</pre>
                    </details>
                  ))}
                </>
              )}
            </section>
          )}
        </>
      )}
    </div>
  );
}

function RuleRow({ rule, api, onChanged, onError }: { rule: Rule; api: Api; onChanged: () => void; onError: (e: string) => void }) {
  const [editing, setEditing] = useState(false);

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
      onChanged();
    } catch (e) {
      onError(errorText(e));
    }
  }

  if (editing) {
    return (
      <RuleForm
        initial={rule}
        submitLabel="Save"
        onCancel={() => setEditing(false)}
        onSubmit={async (patch) => {
          await act(() => api.updateRule(rule.id, patch));
          setEditing(false);
        }}
      />
    );
  }
  return (
    <div className={`rule ${rule.enabled ? "" : "off"}`}>
      <label className="switch" title={rule.enabled ? "On" : "Off"}>
        <input type="checkbox" checked={rule.enabled} onChange={(e) => void act(() => api.updateRule(rule.id, { enabled: e.target.checked }))} />
      </label>
      <span>
        <strong>{rule.name}</strong>
        <span className="muted small"> · {APPLIES[rule.applies]}</span>
        <span className="rule-text">{rule.text}</span>
      </span>
      <span className="row-actions">
        <button className="link" onClick={() => setEditing(true)}>
          Edit
        </button>
        <button className="link danger" onClick={() => void act(() => api.deleteRule(rule.id))}>
          Delete
        </button>
      </span>
    </div>
  );
}

type Draft = { name: string; text: string; applies: RuleApplies };

function NewRule({ onAdd }: { onAdd: (rule: Draft) => Promise<void> }) {
  const [open, setOpen] = useState(false);
  if (!open) {
    return (
      <button className="secondary add-voice-button" onClick={() => setOpen(true)}>
        + Add a rule
      </button>
    );
  }
  return (
    <RuleForm
      submitLabel="Add rule"
      examples
      onCancel={() => setOpen(false)}
      onSubmit={async (rule) => {
        await onAdd(rule);
        setOpen(false);
      }}
    />
  );
}

function RuleForm({
  initial,
  submitLabel,
  examples,
  onSubmit,
  onCancel,
}: {
  initial?: Draft;
  submitLabel: string;
  examples?: boolean;
  onSubmit: (rule: Draft) => Promise<void>;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name ?? "");
  const [text, setText] = useState(initial?.text ?? "");
  const [applies, setApplies] = useState<RuleApplies>(initial?.applies ?? "always");
  const [saving, setSaving] = useState(false);

  return (
    <form
      className="rule-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        try {
          await onSubmit({ name: name.trim(), text: text.trim(), applies });
        } finally {
          setSaving(false);
        }
      }}
    >
      <input placeholder="Name, e.g. No force-pushing" value={name} maxLength={100} onChange={(e) => setName(e.target.value)} autoFocus />
      <textarea
        placeholder="What Amy should always (or never) do"
        value={text}
        maxLength={5000}
        rows={3}
        onChange={(e) => setText(e.target.value)}
      />
      {examples && !text && (
        <p className="muted small">
          For example:{" "}
          {EXAMPLES.map((ex, i) => (
            <button key={i} type="button" className="link small rule-example" onClick={() => setText(ex)}>
              “{ex}”
            </button>
          ))}
        </p>
      )}
      <div className="rule-form-row">
        <select value={applies} onChange={(e) => setApplies(e.target.value as RuleApplies)}>
          {(Object.keys(APPLIES) as RuleApplies[]).map((a) => (
            <option key={a} value={a}>
              {APPLIES[a]}
            </option>
          ))}
        </select>
        <button className="primary" disabled={saving || !name.trim() || !text.trim()}>
          {submitLabel}
        </button>
        <button type="button" className="link" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
