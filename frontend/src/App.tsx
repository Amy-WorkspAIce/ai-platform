import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, AuthError, getApiKey, setApiKey } from "./api";
import { ChatView } from "./views/ChatView";
import { ModelsView } from "./views/ModelsView";

type View = "chat" | "models";

export function App() {
  const [key, setKey] = useState(getApiKey);
  const api = useMemo(() => new Api(key), [key]);
  const [models, setModels] = useState<string[] | null>(null);
  const [error, setError] = useState("");
  const [view, setView] = useState<View>("chat");
  const [chatId, setChatId] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);

  const logout = useCallback(() => {
    setApiKey("");
    setKey("");
    setModels(null);
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
      setError("");
    } catch (e) {
      if (e instanceof AuthError) logout();
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api, logout]);

  useEffect(() => {
    if (key) void refreshModels();
  }, [key, refreshModels]);

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

  return (
    <div className="shell">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand">
          <span className="logo">A</span> Amy
        </div>
        <button
          className="new-chat"
          onClick={() => {
            setChatId((n) => n + 1);
            go("chat");
          }}
        >
          + New chat
        </button>
        <nav>
          <button className={view === "chat" ? "active" : ""} onClick={() => go("chat")}>
            Chat
          </button>
          <button className={view === "models" ? "active" : ""} onClick={() => go("models")}>
            Models
          </button>
        </nav>
        <div className="sidebar-footer">
          <button className="link" onClick={logout}>
            Change API key
          </button>
        </div>
      </aside>
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <div className="main">
        <button className="menu-toggle" aria-label="Menu" onClick={() => setMenuOpen((o) => !o)}>
          ☰
        </button>
        {error && <p className="error banner">{error}</p>}
        {view === "chat" ? (
          <ChatView
            key={chatId}
            api={api}
            models={models}
            onBrowseModels={() => go("models")}
          />
        ) : (
          <ModelsView api={api} onModelsChanged={refreshModels} onUseModel={() => go("chat")} />
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
