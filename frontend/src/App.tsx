import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, AuthError, getApiKey, setApiKey, type Conversation, type ModelInfo } from "./api";
import { loadModelTable } from "./modelTable";
import { ChatView } from "./views/ChatView";
import { LibraryView } from "./views/LibraryView";
import { ModelsView } from "./views/ModelsView";
import { SettingsView } from "./views/SettingsView";

export type View = "chat" | "library" | "models" | "settings";

export function App() {
  const [key, setKey] = useState(getApiKey);
  const api = useMemo(() => new Api(key), [key]);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [error, setError] = useState("");
  const [view, setView] = useState<View>("chat");
  // chatKey remounts ChatView on explicit navigation only, so a chat that saves itself
  // mid-stream (getting an id) isn't remounted.
  const [chat, setChat] = useState<{ key: number; id: string | null }>({ key: 0, id: null });
  const [menuOpen, setMenuOpen] = useState(false);
  const [version, setVersion] = useState("");

  useEffect(() => {
    fetch("/health")
      .then((r) => r.json())
      .then((h) => setVersion(h.version ?? ""))
      .catch(() => {});
  }, []);

  const logout = useCallback(() => {
    setApiKey("");
    setKey("");
    setModels(null);
  }, []);

  const handleError = useCallback(
    (e: unknown) => {
      if (e instanceof AuthError) logout();
      setError(e instanceof Error ? e.message : String(e));
    },
    [logout],
  );

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
      setError("");
    } catch (e) {
      handleError(e);
    }
  }, [api, handleError]);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.conversations());
    } catch (e) {
      handleError(e);
    }
  }, [api, handleError]);

  useEffect(() => {
    if (!key) return;
    void refreshModels();
    void refreshConversations();
    // Prefetch the model table so the Models page opens instantly.
    loadModelTable(api).catch(() => undefined);
  }, [key, api, refreshModels, refreshConversations]);

  if (!key) {
    return (
      <KeyPrompt
        error={error}
        onSubmit={(k) => {
          setApiKey(k);
          setKey(k);
        }}
      />
    );
  }

  const go = (v: View) => {
    setView(v);
    setMenuOpen(false);
  };
  const openChat = (id: string | null) => {
    setChat((c) => ({ key: c.key + 1, id }));
    go("chat");
  };

  const byCapability = (cap: ModelInfo["capability"]) =>
    models?.filter((m) => m.capability === cap) ?? null;

  async function rename(c: Conversation) {
    const title = prompt("Rename chat", c.title)?.trim();
    if (!title || title === c.title) return;
    try {
      await api.renameConversation(c.id, title);
      void refreshConversations();
    } catch (e) {
      handleError(e);
    }
  }

  async function remove(c: Conversation) {
    if (!confirm(`Delete "${c.title}"?`)) return;
    try {
      await api.deleteConversation(c.id);
      if (chat.id === c.id) openChat(null);
      void refreshConversations();
    } catch (e) {
      handleError(e);
    }
  }

  return (
    <div className="shell">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand">
          <span className="logo">A</span> Amy
        </div>
        <button className="new-chat" onClick={() => openChat(null)}>
          + New chat
        </button>
        <nav>
          {(["chat", "library", "models", "settings"] as View[]).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => go(v)}>
              {v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        {conversations.length > 0 && (
          <div className="history">
            <div className="history-label">Recent chats</div>
            {conversations.map((c) => (
              <div
                key={c.id}
                className={`history-item ${view === "chat" && chat.id === c.id ? "active" : ""}`}
              >
                <button className="history-title" title={c.title} onClick={() => openChat(c.id)}>
                  {c.title}
                </button>
                <span className="history-actions">
                  <button aria-label="Rename" title="Rename" onClick={() => rename(c)}>
                    ✎
                  </button>
                  <button aria-label="Delete" title="Delete" onClick={() => remove(c)}>
                    ✕
                  </button>
                </span>
              </div>
            ))}
          </div>
        )}
        <div className="sidebar-footer">
          <button className="link" onClick={logout}>
            Change API key
          </button>
          {version && <span className="version">v{version}</span>}
        </div>
      </aside>
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <div className="main">
        <button className="menu-toggle" aria-label="Menu" onClick={() => setMenuOpen((o) => !o)}>
          ☰
        </button>
        {error && (
          <p className="error banner" onClick={() => setError("")}>
            {error}
          </p>
        )}
        {view === "chat" && (
          <ChatView
            key={chat.key}
            api={api}
            conversationId={chat.id}
            models={byCapability("chat")}
            sttModel={byCapability("transcription")?.[0]?.id}
            ttsModel={byCapability("speech")?.[0]?.id}
            onSaved={(id) => {
              setChat((c) => ({ ...c, id }));
              void refreshConversations();
            }}
            onBrowseModels={() => go("models")}
          />
        )}
        {view === "library" && <LibraryView api={api} onNewChat={() => openChat(null)} />}
        {view === "settings" && <SettingsView api={api} />}
        {view === "models" && (
          <ModelsView
            api={api}
            onModelsChanged={refreshModels}
            onUseModel={() => openChat(null)}
            onOpenSettings={() => go("settings")}
          />
        )}
      </div>
    </div>
  );
}

function KeyPrompt({ error, onSubmit }: { error: string; onSubmit: (key: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <form
      className="key-prompt"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onSubmit(value.trim());
      }}
    >
      <div className="brand big">
        <span className="logo">A</span> Amy
      </div>
      <p className="muted">
        Enter your API key. On first start it is printed in the container logs and saved to{" "}
        <code>/data/secrets/api_key</code>.
      </p>
      <input
        type="password"
        autoFocus
        placeholder="amy-..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button type="submit" className="primary">
        Continue
      </button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}
