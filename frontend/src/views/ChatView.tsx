import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Api, getPref, setPref, type ChatMessage } from "../api";

type Props = {
  api: Api;
  models: string[] | null;
  onBrowseModels: () => void;
};

export function ChatView({ api, models, onBrowseModels }: Props) {
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  // Pick the last-used model if still installed, else the first one.
  useEffect(() => {
    if (!models?.length) return;
    if (!models.includes(model)) {
      const last = getPref("model");
      setModel(models.includes(last) ? last : models[0]);
    }
  }, [models, model]);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and React
    // would treat a returned value as the effect's cleanup function.
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => () => abort.current?.abort(), []);

  const noModels = models !== null && models.length === 0;

  async function send() {
    const text = input.trim();
    if (!text || !model || busy) return;
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "", reasoning: "" }]);
    setInput("");
    setBusy(true);
    setError("");
    abort.current = new AbortController();
    try {
      await api.streamChat(
        model,
        history,
        ({ content, reasoning }) =>
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [
              ...prev.slice(0, -1),
              {
                ...last,
                content: last.content + (content ?? ""),
                reasoning: (last.reasoning ?? "") + (reasoning ?? ""),
              },
            ];
          }),
        abort.current.signal,
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  }

  const composer = (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        void send();
      }}
    >
      <textarea
        value={input}
        rows={1}
        disabled={noModels}
        placeholder={noModels ? "Install a model to start chatting" : "Message Amy…"}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void send();
          }
        }}
      />
      {busy ? (
        <button type="button" className="primary" onClick={() => abort.current?.abort()}>
          Stop
        </button>
      ) : (
        <button type="submit" className="primary" disabled={!input.trim() || !model}>
          Send
        </button>
      )}
    </form>
  );

  return (
    <div className="chat">
      <div className="chat-header">
        {models && models.length > 0 && (
          <select
            className="model-select"
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setPref("model", e.target.value);
            }}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
      </div>

      {messages.length === 0 ? (
        <div className="welcome">
          <h1>How can I help?</h1>
          {noModels ? (
            <div className="welcome-card">
              <p>
                You don't have any models yet. Amy can check your hardware and suggest models
                that will run well on it.
              </p>
              <button className="primary" onClick={onBrowseModels}>
                Find a model for my hardware
              </button>
            </div>
          ) : (
            composer
          )}
        </div>
      ) : (
        <>
          <div className="messages">
            {messages.map((m, i) => (
              <Message key={i} message={m} streaming={busy && i === messages.length - 1} />
            ))}
            {error && <p className="error">{error}</p>}
            <div ref={bottom} />
          </div>
          <div className="composer-dock">{composer}</div>
        </>
      )}
    </div>
  );
}

/** Split out <think>…</think> blocks that some models emit inline. */
function splitThinking(content: string): { thinking: string; answer: string } {
  const match = content.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (!match) return { thinking: "", answer: content };
  return { thinking: match[1].trim(), answer: match[3].trimStart() };
}

function Message({ message, streaming }: { message: ChatMessage; streaming: boolean }) {
  if (message.role === "user") return <div className="msg user">{message.content}</div>;

  const inline = splitThinking(message.content);
  const thinking = [message.reasoning, inline.thinking].filter(Boolean).join("\n").trim();
  const answer = inline.answer;
  const stillThinking = streaming && !answer;

  return (
    <div className="msg assistant">
      {thinking && (
        <details className="thinking" open={stillThinking}>
          <summary>{stillThinking ? "Thinking…" : "Thought process"}</summary>
          <div className="thinking-body">{thinking}</div>
        </details>
      )}
      {answer ? (
        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        </div>
      ) : (
        streaming && !thinking && <span className="typing">●●●</span>
      )}
    </div>
  );
}
