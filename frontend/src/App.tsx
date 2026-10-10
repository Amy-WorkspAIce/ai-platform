import { useCallback, useEffect, useRef, useState } from "react";
import { AuthError, getApiKey, listModels, setApiKey, streamChat, type ChatMessage } from "./api";

export function App() {
  const [key, setKey] = useState(getApiKey);
  const [models, setModels] = useState<string[] | null>(null);
  const [error, setError] = useState("");

  const loadModels = useCallback(async (k: string) => {
    setError("");
    try {
      setModels(await listModels(k));
    } catch (e) {
      if (e instanceof AuthError) setKey("");
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  useEffect(() => {
    if (key) void loadModels(key);
  }, [key, loadModels]);

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

  return (
    <div className="app">
      <header>
        <strong>Amy</strong> <span className="muted">AI Platform</span>
        <button
          className="link"
          onClick={() => {
            setApiKey("");
            setKey("");
          }}
        >
          Change API key
        </button>
      </header>
      {error && <p className="error">{error}</p>}
      {models && <Chat apiKey={key} models={models} />}
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
      <h1>Amy AI Platform</h1>
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
      <button type="submit">Continue</button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}

function Chat({ apiKey, models }: { apiKey: string; models: string[] }) {
  const [model, setModel] = useState(models[0] ?? "");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function send() {
    const text = input.trim();
    if (!text || !model || busy) return;
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    setError("");
    abort.current = new AbortController();
    try {
      await streamChat(
        apiKey,
        model,
        history,
        (delta) =>
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [...prev.slice(0, -1), { ...last, content: last.content + delta }];
          }),
        abort.current.signal,
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(String(e instanceof Error ? e.message : e));
      }
    } finally {
      setBusy(false);
    }
  }

  if (models.length === 0) {
    return (
      <div className="empty">
        <h2>No models available</h2>
        <p className="muted">
          Pull a model into Ollama (e.g. <code>docker exec -it amy-ai-platform ollama pull llama3.2</code>)
          or configure a provider with <code>AMY_PROVIDERS</code>, then reload.
        </p>
      </div>
    );
  }

  return (
    <main className="chat">
      <div className="toolbar">
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
        <button className="link" onClick={() => setMessages([])} disabled={busy}>
          New chat
        </button>
      </div>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.content || (busy && i === messages.length - 1 ? "…" : "")}
          </div>
        ))}
        {error && <p className="error">{error}</p>}
        <div ref={bottom} />
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea
          value={input}
          placeholder="Message Amy…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        {busy ? (
          <button type="button" onClick={() => abort.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </main>
  );
}
