import { useCallback, useEffect, useState } from "react";
import { Api, type Fact, type MemoryState } from "../api";
import { ago } from "../modelTable";

type Props = { api: Api; onOpenChat: (id: string) => void };

/** What Amy knows about you: view, search, edit, add, forget. */
export function MemoryView({ api, onOpenChat }: Props) {
  const [state, setState] = useState<MemoryState | null>(null);
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [category, setCategory] = useState("all");
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [newFact, setNewFact] = useState("");
  const [newCategory, setNewCategory] = useState("general");
  const [error, setError] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 350);
    return () => clearTimeout(t);
  }, [query]);

  const load = useCallback(async () => {
    try {
      setState(await api.memory(debounced || undefined));
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api, debounced]);

  useEffect(() => {
    void load();
  }, [load]);

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!state) {
    return <div className="page">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  }

  const facts = state.facts.filter((f) => category === "all" || f.category === category);
  const usedCategories = state.categories.filter((c) => state.facts.some((f) => f.category === c));

  return (
    <div className="page memory">
      <h1>Memory</h1>
      <p className="muted small">
        Facts Amy has learned about you from your chats. The relevant ones are brought into new conversations so you
        don't have to repeat yourself. Remove anything you don't want remembered; you can also say "forget that" in
        chat. Passwords, keys and card numbers are never stored.
      </p>

      <section className="card setting">
        <div className="memory-switches">
          <label className="switch">
            <input type="checkbox" checked={state.enabled} onChange={(e) => void act(() => api.memorySettings({ enabled: e.target.checked }))} />
            Use memory in chats
          </label>
          <label className="switch">
            <input
              type="checkbox"
              checked={state.auto}
              disabled={!state.enabled}
              onChange={(e) => void act(() => api.memorySettings({ auto: e.target.checked }))}
            />
            Learn new facts automatically
          </label>
        </div>
        {!state.embedding_model && state.facts.length > 30 && (
          <p className="muted small">
            Tip: install an embedding model (e.g. Nomic Embed Text from Models → Embeddings) for smarter recall and
            search by meaning.
          </p>
        )}
      </section>

      <div className="browse-controls">
        <input
          className="browse-search"
          type="search"
          placeholder={state.embedding_model ? "Search by meaning, e.g. “my hardware”" : "Search facts…"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="muted small">
          {state.facts.length} fact{state.facts.length === 1 ? "" : "s"}
        </span>
      </div>
      {usedCategories.length > 1 && (
        <div className="chips">
          <button className={category === "all" ? "chip active" : "chip"} onClick={() => setCategory("all")}>
            All
          </button>
          {usedCategories.map((c) => (
            <button key={c} className={category === c ? "chip active" : "chip"} onClick={() => setCategory(c)}>
              {c[0].toUpperCase() + c.slice(1)}
            </button>
          ))}
        </div>
      )}
      {error && <p className="error">{error}</p>}

      <div className="card fact-list">
        {facts.length === 0 && (
          <p className="muted">
            {debounced ? "No matching facts." : "Nothing remembered yet. Chat a bit, or add a fact below."}
          </p>
        )}
        {facts.map((f: Fact) => (
          <div key={f.id} className="fact">
            {editing === f.id ? (
              <form
                className="fact-edit"
                onSubmit={(e) => {
                  e.preventDefault();
                  void act(() => api.updateFact(f.id, { text: draft })).then(() => setEditing(null));
                }}
              >
                <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} />
                <button className="primary" disabled={draft.trim().length < 3}>
                  Save
                </button>
                <button type="button" className="link" onClick={() => setEditing(null)}>
                  Cancel
                </button>
              </form>
            ) : (
              <>
                <div className="fact-text">
                  <span>{f.text}</span>
                  <span className="muted small">
                    <span className="tag">{f.category}</span> learned {ago(f.created_at)}
                    {f.updated_at - f.created_at > 60 ? ` · updated ${ago(f.updated_at)}` : ""}
                    {f.source_conversation_id && (
                      <>
                        {" · "}
                        <button className="link small" onClick={() => onOpenChat(f.source_conversation_id!)}>
                          from this chat
                        </button>
                      </>
                    )}
                  </span>
                </div>
                <span className="row-actions">
                  <button
                    className="link"
                    onClick={() => {
                      setEditing(f.id);
                      setDraft(f.text);
                    }}
                  >
                    Edit
                  </button>
                  <button className="link danger" onClick={() => void act(() => api.deleteFact(f.id))}>
                    Forget
                  </button>
                </span>
              </>
            )}
          </div>
        ))}
      </div>

      <h2>Add a fact</h2>
      <form
        className="add-fact"
        onSubmit={(e) => {
          e.preventDefault();
          if (newFact.trim().length >= 3)
            void act(() => api.addFact(newFact.trim(), newCategory)).then(() => setNewFact(""));
        }}
      >
        <input placeholder="e.g. Prefers metric units" value={newFact} onChange={(e) => setNewFact(e.target.value)} />
        <select value={newCategory} onChange={(e) => setNewCategory(e.target.value)} aria-label="Category">
          {state.categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <button className="primary" disabled={newFact.trim().length < 3}>
          Add
        </button>
      </form>

      {state.facts.length > 0 && (
        <button
          className="link danger forget-all"
          onClick={() => {
            if (confirm(`Forget all ${state.facts.length} facts? This can't be undone.`)) void act(() => api.forgetEverything());
          }}
        >
          Forget everything
        </button>
      )}
    </div>
  );
}
