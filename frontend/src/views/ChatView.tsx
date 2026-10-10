import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Api, getPref, setPref, type ChatMessage, type ModelInfo } from "../api";
import { canRecord, extensionFor, useRecorder } from "../recorder";

type Props = {
  api: Api;
  conversationId: string | null;
  models: ModelInfo[] | null;
  sttModel?: string;
  ttsModel?: string;
  onSaved: (id: string) => void;
  onBrowseModels: () => void;
};

export function ChatView({ api, conversationId, models, sttModel, ttsModel, onSaved, onBrowseModels }: Props) {
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(!!conversationId);
  const [error, setError] = useState("");
  const convId = useRef(conversationId);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const ids = models?.map((m) => m.id) ?? [];

  // Load a saved conversation.
  useEffect(() => {
    if (!conversationId) return;
    let cancelled = false;
    api
      .conversation(conversationId)
      .then((c) => {
        if (cancelled) return;
        setMessages(
          c.messages.map((m) => ({ role: m.role, content: m.content, reasoning: m.reasoning ?? "" })),
        );
        if (c.model) setModel(c.model);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [api, conversationId]);

  // Pick the conversation's model if still installed, else the last-used one, else the first.
  useEffect(() => {
    if (!ids.length || ids.includes(model)) return;
    const last = getPref("model");
    setModel(ids.includes(last) ? last : ids[0]);
  }, [ids.join(","), model]);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and React
    // would treat a returned value as the effect's cleanup function.
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => () => abort.current?.abort(), []);

  const noModels = models !== null && models.length === 0;

  async function save(fn: (id: string) => Promise<unknown>) {
    try {
      if (convId.current) await fn(convId.current);
    } catch (e) {
      setError(`Couldn't save the chat: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || !model || busy) return;
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "", reasoning: "" }]);
    setInput("");
    setBusy(true);
    setError("");

    if (!convId.current) {
      try {
        const conv = await api.createConversation(titleFrom(text), model);
        convId.current = conv.id;
        onSaved(conv.id);
      } catch (e) {
        setError(`Couldn't save the chat: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    void save((id) => api.addMessage(id, { role: "user", content: text }));

    let content = "";
    let reasoning = "";
    abort.current = new AbortController();
    try {
      await api.streamChat(
        model,
        history,
        (d) => {
          content += d.content ?? "";
          reasoning += d.reasoning ?? "";
          setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content, reasoning }]);
        },
        abort.current.signal,
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
      if (content || reasoning) {
        await save((id) => api.addMessage(id, { role: "assistant", content, reasoning, model }));
        if (convId.current) onSaved(convId.current);
      }
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
      {sttModel && (
        <MicButton
          api={api}
          model={sttModel}
          onText={(t) => setInput((v) => (v ? `${v} ${t}` : t))}
          onError={setError}
        />
      )}
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
        {ids.length > 0 && (
          <select
            className="model-select"
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setPref("model", e.target.value);
            }}
          >
            {ids.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
      </div>

      {loading ? (
        <div className="welcome muted">Loading…</div>
      ) : messages.length === 0 ? (
        <div className="welcome">
          <h1>How can I help?</h1>
          {noModels ? (
            <div className="welcome-card">
              <p>
                You don't have any chat models yet. Amy can check your hardware and suggest models
                that will run well on it.
              </p>
              <button className="primary" onClick={onBrowseModels}>
                Find a model for my hardware
              </button>
            </div>
          ) : (
            composer
          )}
          {error && <p className="error">{error}</p>}
        </div>
      ) : (
        <>
          <div className="messages">
            {messages.map((m, i) => (
              <Message
                key={i}
                api={api}
                message={m}
                ttsModel={ttsModel}
                streaming={busy && i === messages.length - 1}
              />
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

function titleFrom(text: string): string {
  const line = text.split("\n")[0].trim();
  return line.length > 60 ? `${line.slice(0, 57)}…` : line || "New chat";
}

/** Split out <think>…</think> blocks that some models emit inline. */
function splitThinking(content: string): { thinking: string; answer: string } {
  const match = content.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (!match) return { thinking: "", answer: content };
  return { thinking: match[1].trim(), answer: match[3].trimStart() };
}

/** Rough markdown-to-speech cleanup: drop code blocks and formatting characters. */
function speakable(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " (code omitted) ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[#>*_|~-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function Message({
  api,
  message,
  ttsModel,
  streaming,
}: {
  api: Api;
  message: ChatMessage;
  ttsModel?: string;
  streaming: boolean;
}) {
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
      {answer && !streaming && ttsModel && (
        <div className="msg-actions">
          <SpeakButton api={api} model={ttsModel} text={speakable(answer)} />
        </div>
      )}
    </div>
  );
}

function SpeakButton({ api, model, text }: { api: Api; model: string; text: string }) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => () => audio.current?.pause(), []);

  async function toggle() {
    if (state === "playing") {
      audio.current?.pause();
      setState("idle");
      return;
    }
    setState("loading");
    try {
      const url = URL.createObjectURL(await api.speak(model, text));
      const a = new Audio(url);
      a.onended = () => {
        setState("idle");
        URL.revokeObjectURL(url);
      };
      audio.current = a;
      await a.play();
      setState("playing");
    } catch {
      setState("idle");
    }
  }

  return (
    <button className="icon-button" onClick={toggle} title="Read aloud" disabled={state === "loading"}>
      {state === "loading" ? "…" : state === "playing" ? "■ Stop" : "🔊 Read aloud"}
    </button>
  );
}

function MicButton({
  api,
  model,
  onText,
  onError,
}: {
  api: Api;
  model: string;
  onText: (text: string) => void;
  onError: (msg: string) => void;
}) {
  const { recording, start, stop } = useRecorder();
  const [transcribing, setTranscribing] = useState(false);
  const available = canRecord();

  async function toggle() {
    if (!recording) {
      try {
        await start();
      } catch {
        onError("Couldn't access the microphone.");
      }
      return;
    }
    const blob = await stop();
    if (!blob) return;
    setTranscribing(true);
    try {
      const text = await api.transcribe(model, blob, `dictation.${extensionFor(blob)}`);
      if (text) onText(text);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setTranscribing(false);
    }
  }

  return (
    <button
      type="button"
      className={`icon-button mic ${recording ? "recording" : ""}`}
      disabled={!available || transcribing}
      title={
        available
          ? recording
            ? "Stop and transcribe"
            : "Dictate"
          : "Microphone access needs HTTPS (or localhost)"
      }
      onClick={toggle}
    >
      {transcribing ? "…" : recording ? "● Stop" : "🎤"}
    </button>
  );
}
