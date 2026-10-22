import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, AuthError, getApiKey, setApiKey, type Conversation, type Folder, type ModelInfo } from "./api";
import { ChatList } from "./ChatList";
import { Icon } from "./icons";
import { loadModelTable } from "./modelTable";
import { ChatView } from "./views/ChatView";
import { LibraryView } from "./views/LibraryView";
import { MemoryView } from "./views/MemoryView";
import { ModelsView } from "./views/ModelsView";
import { SettingsView } from "./views/SettingsView";

export type View = "chat" | "library" | "memory" | "models" | "settings";

export function App() {
  const [key, setKey] = useState(getApiKey);
  const api = useMemo(() => new Api(key), [key]);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [error, setError] = useState("");
  const [view, setView] = useState<View>("chat");
  // chatKey remounts ChatView on explicit navigation only, so a chat that saves itself
  // mid-stream (getting an id) isn't remounted.
  const [chat, setChat] = useState<{ key: number; id: string | null; folderId: string | null }>({
    key: 0,
    id: null,
    folderId: null,
  });
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
      const [c, f] = await Promise.all([api.conversations(), api.folders()]);
      setConversations(c);
      setFolders(f);
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
  const openChat = (id: string | null, folderId: string | null = null) => {
    setChat((c) => ({ key: c.key + 1, id, folderId }));
    go("chat");
  };

  const byCapability = (cap: ModelInfo["capability"]) =>
    models?.filter((m) => m.capability === cap) ?? null;

  return (
    <div className="shell">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand">
          <Logo />
          {version && (
            <span className="version" title={`AI Platform ${version}`}>
              v{version}
            </span>
          )}
        </div>
        <button className="new-chat" onClick={() => openChat(null)}>
          + New chat
        </button>
        <nav>
          {(["chat", "library", "memory", "models", "settings"] as View[]).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => go(v)}>
              {v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        <ChatList
          api={api}
          conversations={conversations}
          folders={folders}
          activeId={view === "chat" ? chat.id : null}
          onOpen={(id) => openChat(id)}
          onNewChat={(folderId) => openChat(null, folderId)}
          onChanged={refreshConversations}
          onDeleted={(id) => chat.id === id && openChat(null)}
          onError={handleError}
        />
      </aside>
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <div className="main">
        <button className="menu-toggle" aria-label="Menu" onClick={() => setMenuOpen((o) => !o)}>
          <Icon name="menu" size={20} />
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
            folderId={chat.folderId}
            folderName={folders.find((f) => f.id === chat.folderId)?.name}
            onOpenMemory={() => go("memory")}
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
        {view === "memory" && <MemoryView api={api} onOpenChat={(id) => openChat(id)} />}
        {view === "settings" && <SettingsView api={api} onSignOut={logout} />}
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
        <Logo />
        <p className="readout">AI Platform</p>
      </div>
      <p className="muted">
        Installed from the Amy-WorkspAIce hub? Use its <strong>Open</strong> button and you're signed in automatically. Otherwise
        enter your API key: it's printed in the container logs on first start and saved to <code>/data/secrets/api_key</code>.
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

/** The brand lockup; the light-ground version shows in the light theme. */
function Logo() {
  return (
    <>
      <img className="brand-logo on-dark" src="/brand/amy-workspaice-lockup.svg" alt="Amy-WorkspAIce" />
      <img className="brand-logo on-light" src="/brand/amy-workspaice-lockup-light.svg" alt="Amy-WorkspAIce" />
    </>
  );
}
