import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Api,
  getPref,
  setPref,
  type AgentEvent,
  type Attachment,
  type ChatDetails as Details,
  type ModelInfo,
  type StoredMessage,
  type WorkspaceInfo,
} from "../api";
import { AttachmentView, ImageView } from "../media";
import { routeAudio } from "../devices";
import { canRecord, extensionFor, useRecorder } from "../recorder";
import { Icon } from "../icons";
import { ChatDetails, type Focus, type Section, type Turn } from "./ChatDetails";

const AUTO = "auto";

type Props = {
  api: Api;
  conversationId: string | null;
  folderId?: string | null;
  folderName?: string;
  onOpenMemory?: () => void;
  models: ModelInfo[] | null;
  sttModel?: string;
  ttsModel?: string;
  onSaved: (id: string) => void;
  onBrowseModels: () => void;
  onOpenRules?: (scope: string) => void;
};

type ToolStatus = "awaiting" | "running" | "ok" | "error";

type Item =
  | { kind: "user"; content: string }
  | { kind: "assistant"; content: string; reasoning: string }
  | { kind: "notice"; text: string; memory?: boolean }
  | {
      kind: "tool";
      id: string;
      name: string;
      summary: string;
      args: Record<string, unknown> | string;
      status: ToolStatus;
      output?: string;
      attachments?: Attachment[];
    };

type Mode = "chat" | "agent";

const SUMMARY_ARG: Record<string, string> = {
  web_search: "query",
  fetch_url: "url",
  list_files: "path",
  read_file: "path",
  search_files: "pattern",
  write_file: "path",
  edit_file: "path",
  run_command: "command",
  generate_image: "prompt",
  edit_image: "instruction",
  generate_video: "prompt",
  speak: "text",
  ask_models: "question",
  recall_memory: "query",
};

/** Tools whose result is media shown in the chat. */
const MEDIA_TOOLS = new Set(["generate_image", "edit_image", "generate_video", "speak"]);

const TOOL_LABEL: Record<string, string> = {
  web_search: "Searched the web",
  fetch_url: "Read page",
  list_files: "Listed files",
  read_file: "Read file",
  search_files: "Searched files",
  write_file: "Write file",
  edit_file: "Edit file",
  run_command: "Run command",
  generate_image: "Image",
  edit_image: "Edit",
  generate_video: "Video",
  speak: "Voice",
  ask_models: "Asked other models",
  recall_memory: "Looked in memory",
};

// Images the user attached are referenced in their message so the model can edit them.
const ATTACHED = /\[Attached image: image_id=([a-f0-9]+)(?: \(([^)]*)\))?\]/g;

function attachedIds(text: string): string[] {
  return [...text.matchAll(ATTACHED)].map((m) => m[1]);
}

function parseArgs(raw: string): Record<string, unknown> | string {
  try {
    const v = JSON.parse(raw || "{}");
    return v && typeof v === "object" ? v : raw;
  } catch {
    return raw;
  }
}

function summaryOf(name: string, args: Record<string, unknown> | string): string {
  const key = SUMMARY_ARG[name];
  return typeof args === "object" && key && args[key] != null ? String(args[key]) : "";
}

/** Rebuild display items (including tool cards) from saved messages. */
function itemsFrom(messages: StoredMessage[]): Item[] {
  const items: Item[] = [];
  for (const m of messages) {
    if (m.role === "user") items.push({ kind: "user", content: m.content });
    else if (m.role === "assistant") {
      if (m.content || m.reasoning) {
        items.push({ kind: "assistant", content: m.content, reasoning: m.reasoning ?? "" });
      }
      for (const call of m.meta?.tool_calls ?? []) {
        const args = parseArgs(call.function.arguments);
        items.push({
          kind: "tool",
          id: call.id,
          name: call.function.name,
          args,
          summary: summaryOf(call.function.name, args),
          status: "error",
          output: "(no result recorded)",
        });
      }
    } else if (m.role === "tool") {
      const tool = items.find((i) => i.kind === "tool" && i.id === m.meta?.tool_call_id);
      if (tool && tool.kind === "tool") {
        tool.status = m.meta?.ok ? "ok" : "error";
        tool.output = m.content;
        tool.attachments = m.meta?.attachments;
      }
    }
  }
  return items;
}

/** Each user message with the thinking and tool use behind its answer (the Details panel). */
function turnsFrom(items: Item[]): Turn[] {
  const turns: Turn[] = [];
  for (const item of items) {
    if (item.kind === "user") {
      turns.push({ prompt: item.content.replace(ATTACHED, "").trim().slice(0, 140), steps: [] });
      continue;
    }
    const turn = turns[turns.length - 1];
    if (!turn) continue;
    if (item.kind === "assistant") {
      const text = thinkingOf(item.content, item.reasoning);
      if (text) turn.steps.push({ kind: "thought", text });
    } else if (item.kind === "tool") {
      const ok = item.status === "ok" ? true : item.status === "error" ? false : null;
      turn.steps.push({ kind: "tool", label: TOOL_LABEL[item.name] ?? item.name, summary: item.summary, ok });
    }
  }
  return turns;
}

export function ChatView({
  api,
  conversationId,
  folderId,
  folderName,
  onOpenMemory,
  models,
  sttModel,
  ttsModel,
  onSaved,
  onBrowseModels,
  onOpenRules,
}: Props) {
  const [model, setModel] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(!!conversationId);
  const [error, setError] = useState("");
  const [mode, setMode] = useState<Mode>((getPref("mode") as Mode) || "chat");
  const [pendingImages, setPendingImages] = useState<{ id: string; name: string }[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const [workspace, setWorkspace] = useState(getPref("workspace"));
  const [autoApprove, setAutoApprove] = useState(false);
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[] | null>(null);
  const convId = useRef(conversationId);
  // A chat this view started: already on screen (mid-stream), so it isn't reloaded.
  const created = useRef<string | null>(null);
  const runId = useRef<string | null>(null);
  const sendCount = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  // The model the server picked last on Auto, so the chat only notes when it changes.
  const autoPick = useRef("");
  // The Details panel (hidden by default; stays open across chats once opened).
  // On phones it covers the chat, so it always starts hidden there.
  const [detailsOpen, setDetailsOpen] = useState(() => getPref("chatDetails") === "open" && window.innerWidth > 860);
  const [details, setDetails] = useState<Details | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  const ids = models?.map((m) => m.id) ?? [];
  const modelInfo = models?.find((m) => m.id === model);

  function loadDetails() {
    const id = convId.current;
    if (id) api.conversationDetails(id).then(setDetails).catch(() => {});
  }

  function showDetails(section: Section, turn?: number) {
    setDetailsOpen(true);
    setPref("chatDetails", "open");
    setFocus((f) => ({ section, turn, nonce: (f?.nonce ?? 0) + 1 }));
  }

  function hideDetails() {
    setDetailsOpen(false);
    setPref("chatDetails", "closed");
  }

  async function forget(memoryId: string) {
    try {
      await api.deleteFact(memoryId);
      loadDetails();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function upsertSubagent(agent: Details["subagents"][number]) {
    setDetails((d) => {
      const base = d ?? { memories: [], subagents: [] };
      const others = base.subagents.filter((a) => a.id !== agent.id);
      return { ...base, subagents: [...others, agent].sort((a, b) => a.started_at - b.started_at) };
    });
  }

  useEffect(loadDetails, [api, conversationId]);

  // Load a saved conversation.
  useEffect(() => {
    if (!conversationId || conversationId === created.current) return;
    let cancelled = false;
    api
      .conversation(conversationId)
      .then((c) => {
        if (cancelled) return;
        setItems(itemsFrom(c.messages));
        if (c.model) setModel(c.model);
        if (c.settings?.mode) setMode(c.settings.mode);
        if (c.settings?.workspace) setWorkspace(c.settings.workspace);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [api, conversationId]);

  useEffect(() => {
    if (!ids.length || model === AUTO || ids.includes(model)) return;
    const last = getPref("model") || AUTO;
    setModel(last === AUTO || ids.includes(last) ? last : AUTO);
  }, [ids.join(","), model]);

  const refreshWorkspaces = () =>
    api
      .workspaces()
      .then(setWorkspaces)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));

  useEffect(() => {
    if (mode === "agent" && workspaces === null) void refreshWorkspaces();
  }, [mode]);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and React
    // would treat a returned value as the effect's cleanup function.
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);
  useEffect(() => () => abort.current?.abort(), []);

  const noModels = models !== null && models.length === 0;
  const needsWorkspace = mode === "agent" && !workspace;

  function appendText(field: "content" | "reasoning", delta: string) {
    setItems((prev) => {
      const last = prev[prev.length - 1];
      if (last?.kind === "assistant") {
        return [...prev.slice(0, -1), { ...last, [field]: last[field] + delta }];
      }
      return [...prev, { kind: "assistant", content: "", reasoning: "", [field]: delta }];
    });
  }

  function updateTool(id: string, patch: Partial<Extract<Item, { kind: "tool" }>>) {
    setItems((prev) => prev.map((i) => (i.kind === "tool" && i.id === id ? { ...i, ...patch } : i)));
  }

  function onEvent(e: AgentEvent) {
    switch (e.type) {
      case "model": {
        if (e.model === autoPick.current) break;
        const first = !autoPick.current;
        autoPick.current = e.model;
        const why = e.purpose === "coding" ? "the coding model" : "the chat model";
        if (!first || e.purpose === "coding") {
          setItems((prev) => [...prev, { kind: "notice", text: `Auto: answering with ${e.model.split("/").pop()}, ${why}` }]);
        }
        break;
      }
      case "start":
        runId.current = e.run_id;
        if (!convId.current) {
          convId.current = created.current = e.conversation_id;
          onSaved(e.conversation_id);
        }
        loadDetails(); // memories recalled for this message
        break;
      case "text":
        appendText("content", e.delta);
        break;
      case "reasoning":
        appendText("reasoning", e.delta);
        break;
      case "tool_call":
        setItems((prev) => [
          ...prev,
          {
            kind: "tool",
            id: e.id,
            name: e.name,
            args: e.arguments,
            summary: e.summary,
            status: e.needs_approval ? "awaiting" : "running",
          },
        ]);
        break;
      case "tool_result":
        updateTool(e.id, { status: e.ok ? "ok" : "error", output: e.output, attachments: e.attachments });
        break;
      case "notice":
        setItems((prev) => [...prev, { kind: "notice", text: e.message }]);
        break;
      case "answer_done":
        // The reply is complete; memory may still be updating in the background.
        setBusy(false);
        break;
      case "memory": {
        const verb = { remembered: "Remembered", updated: "Updated memory", removed: "Forgot", failed: "Couldn't update memory", recalled: "Recalled" }[e.action];
        setItems((prev) => [...prev, { kind: "notice", text: `${verb}: ${e.text}`, memory: true }]);
        loadDetails();
        break;
      }
      case "subagent": {
        const { type: _, ...agent } = e;
        upsertSubagent(agent);
        break;
      }
      case "error":
        setError(e.message);
        break;
    }
  }

  async function decide(id: string, approve: boolean, always = false) {
    if (!runId.current) return;
    try {
      await api.approve(runId.current, id, approve, always);
      updateTool(id, { status: approve ? "running" : "error", output: approve ? undefined : "Denied." });
      if (always && approve) setAutoApprove(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function sendWithTools(text: string) {
    await api.agentChat(
      {
        model,
        message: text,
        conversation_id: convId.current,
        mode,
        workspace: mode === "agent" ? workspace : null,
        auto_approve: autoApprove,
        folder_id: convId.current ? undefined : folderId,
      },
      onEvent,
      abort.current?.signal,
    );
    if (convId.current) onSaved(convId.current);
  }

  async function attachFiles(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of Array.from(files)) {
        if (file.type.startsWith("image/")) {
          const id = await api.uploadImage(file);
          setPendingImages((p) => [...p, { id, name: file.name }]);
        } else if (file.type.startsWith("audio/") || file.type.startsWith("video/")) {
          if (!sttModel) throw new Error("Install a speech-to-text model (e.g. Whisper) to attach audio.");
          const text = await api.transcribe(sttModel, file, file.name);
          setInput((v) => `${v ? `${v}\n\n` : ""}Transcript of ${file.name}:\n${text}`);
        } else {
          throw new Error(`${file.name}: only images and audio can be attached.`);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function send() {
    const typed = input.trim();
    if ((!typed && !pendingImages.length) || !model || busy || needsWorkspace) return;
    const notes = pendingImages.map((p) => `[Attached image: image_id=${p.id} (${p.name.replace(/[()\]]/g, "")})]`);
    const text = [typed || "Here's an image.", ...notes].join("\n\n");
    setPendingImages([]);
    setItems((prev) => [...prev, { kind: "user", content: text }]);
    setInput("");
    setBusy(true);
    setError("");
    abort.current = new AbortController();
    // A reply finishes (answer_done) before its stream closes (memory updates last), so a
    // new message may start meanwhile; only the latest send may reset shared state.
    const mine = ++sendCount.current;
    try {
      // All chats go through the server: it adds the system prompt, saves the chat, and offers
      // the tools (web, image/video/voice delegation, and workspace tools in agent mode).
      await sendWithTools(text);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      loadDetails();
      if (sendCount.current !== mine) return;
      setBusy(false);
      runId.current = null;
      setItems((prev) =>
        prev.map((i) => (i.kind === "tool" && (i.status === "running" || i.status === "awaiting") ? { ...i, status: "error", output: i.output ?? "Stopped." } : i)),
      );
    }
  }

  const toolbar = (
    <div className="chat-toolbar">
      <div className="segmented small">
        {(["chat", "agent"] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            className={mode === m ? "active" : ""}
            disabled={busy}
            onClick={() => {
              setMode(m);
              setPref("mode", m);
            }}
          >
            {m === "chat" ? "Chat" : "Agent"}
          </button>
        ))}
      </div>
      {mode === "agent" && (
        <>
          <WorkspacePicker
            api={api}
            value={workspace}
            workspaces={workspaces}
            disabled={busy}
            onChange={(w) => {
              setWorkspace(w);
              setPref("workspace", w);
            }}
            onCreated={async (name) => {
              await refreshWorkspaces();
              setWorkspace(name);
              setPref("workspace", name);
            }}
          />
          {workspace && onOpenRules && (
            <button type="button" className="link small" title="Rules Amy follows in this workspace" onClick={() => onOpenRules(`workspace:${workspace}`)}>
              Rules
            </button>
          )}
          <label className="auto-approve" title="Run commands and edit files without asking">
            <input type="checkbox" checked={autoApprove} onChange={(e) => setAutoApprove(e.target.checked)} />
            Auto-approve
          </label>
        </>
      )}
      {modelInfo?.tools === false && (
        <span className="warn small">
          This model can't use tools (web search, images, video). Pick one tagged "Tools" for those.
        </span>
      )}
    </div>
  );

  const composer = (
    <div className="composer-wrap">
      {toolbar}
      {pendingImages.length > 0 && (
        <div className="pending-attachments">
          {pendingImages.map((p) => (
            <span key={p.id} className="pending-chip">
              <Icon name="image" /> {p.name}
              <button type="button" aria-label="Remove" onClick={() => setPendingImages((x) => x.filter((y) => y.id !== p.id))}>
                <Icon name="close" size={14} />
              </button>
            </span>
          ))}
        </div>
      )}
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
          placeholder={
            noModels
              ? "Install a model to start chatting"
              : needsWorkspace
                ? "Choose or create a workspace for the agent first"
                : mode === "agent"
                  ? "Describe a change or feature…"
                  : "Message Amy…"
          }
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <input
          ref={fileInput}
          type="file"
          hidden
          multiple
          accept="image/*,audio/*"
          onChange={(e) => void attachFiles(e.target.files)}
        />
        <button
          type="button"
          className="icon-button"
          title="Attach an image (to edit or discuss) or an audio file (to transcribe)"
          disabled={uploading || noModels}
          onClick={() => fileInput.current?.click()}
        >
          {uploading ? "…" : <Icon name="attach" size={18} label="Attach image" />}
        </button>
        <MicButton api={api} model={sttModel} onText={(t) => setInput((v) => (v ? `${v} ${t}` : t))} onError={setError} />
        {busy ? (
          <button type="button" className="primary" onClick={() => abort.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="primary" disabled={(!input.trim() && !pendingImages.length) || !model || needsWorkspace}>
            Send
          </button>
        )}
      </form>
    </div>
  );

  const lastIndex = items.length - 1;
  const turns = turnsFrom(items);
  const running = details?.subagents.filter((a) => a.status === "running").length ?? 0;
  const lastItem = items[lastIndex];
  const thinkingNow = busy && lastItem?.kind === "assistant" && !splitThinking(lastItem.content).answer && !!thinkingOf(lastItem.content, lastItem.reasoning);
  // Which turn each item belongs to, so "Thought process" opens the right one.
  let turnIndex = -1;
  const turnOf = items.map((item) => (item.kind === "user" ? ++turnIndex : turnIndex));

  return (
    <div className="chat-area">
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
            <option value={AUTO}>Auto: picks the right model</option>
            {models?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
                {m.tools ? "  · tools" : ""}
              </option>
            ))}
          </select>
        )}
        {!detailsOpen && (
          <button
            type="button"
            className="icon-button details-toggle"
            onClick={() => showDetails(running ? "subagents" : "memory")}
            title="Details: memory, thought process and sub-agents for this chat"
          >
            <Icon name="panel" size={18} label="Show details" />
            {running > 0 && <span className="tool-status running" aria-label={`${running} sub-agents running`} />}
          </button>
        )}
      </div>

      {loading ? (
        <div className="welcome muted">Loading…</div>
      ) : items.length === 0 ? (
        <div className="welcome">
          <h1>{mode === "agent" ? "What should I build?" : "How can I help?"}</h1>
          {folderName && <p className="muted small">New chat in <Icon name="folder" size={14} /> {folderName}</p>}
          {noModels ? (
            <div className="welcome-card">
              <p>
                You don't have any chat models yet. Amy can check your hardware and suggest models that will run
                well on it.
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
            {items.map((item, i) =>
              item.kind === "user" ? (
                <UserMessage key={i} api={api} content={item.content} />
              ) : item.kind === "notice" ? (
                <p key={i} className={`notice muted small ${item.memory ? "memory-note" : ""}`}>
                  {item.memory ? (
                    <button className="link" onClick={() => showDetails("memory")} title="See what Amy remembers from this chat">
                      <Icon name="orbit" size={14} /> {item.text}
                    </button>
                  ) : (
                    item.text
                  )}
                </p>
              ) : item.kind === "assistant" ? (
                <AssistantMessage
                  key={i}
                  api={api}
                  content={item.content}
                  reasoning={item.reasoning}
                  ttsModel={ttsModel}
                  streaming={busy && i === lastIndex}
                  onShowThinking={() => showDetails("thinking", turnOf[i])}
                />
              ) : (
                <ToolCard key={item.id} api={api} item={item} onDecide={decide} />
              ),
            )}
            {busy && items[lastIndex]?.kind === "user" && <span className="typing">●●●</span>}
            {error && <p className="error">{error}</p>}
            <div ref={bottom} />
          </div>
          <div className="composer-dock">{composer}</div>
        </>
      )}
    </div>
    {detailsOpen && (
      <>
        <div className="details-scrim" onClick={hideDetails} />
        <ChatDetails
          details={details}
          turns={turns}
          thinkingNow={thinkingNow}
          focus={focus}
          onClose={hideDetails}
          onForget={(id) => void forget(id)}
          onOpenMemory={onOpenMemory}
        />
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

/** Everything a reply thought: streamed reasoning plus any inline <think> block. */
function thinkingOf(content: string, reasoning: string): string {
  return [reasoning, splitThinking(content).thinking].filter(Boolean).join("\n").trim();
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

function AssistantMessage({
  api,
  content,
  reasoning,
  ttsModel,
  streaming,
  onShowThinking,
}: {
  api: Api;
  content: string;
  reasoning: string;
  ttsModel?: string;
  streaming: boolean;
  onShowThinking: () => void;
}) {
  const thinking = thinkingOf(content, reasoning);
  const answer = splitThinking(content).answer;
  const stillThinking = streaming && !answer;

  // The thinking itself lives in the Details panel; the reply only links to it.
  const link = thinking && (
    <button type="button" className={`link thinking-link ${stillThinking ? "typing" : ""}`} onClick={onShowThinking} title="Read the thought process in Details">
      {stillThinking ? "Thinking…" : "Thought process"}
    </button>
  );
  // A step that only thought (before using a tool) is just the link, not an empty reply.
  if (thinking && !answer) return <div className="msg-thinking">{link}</div>;

  return (
    <div className="msg assistant">
      {link}
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

function UserMessage({ api, content }: { api: Api; content: string }) {
  const ids = attachedIds(content);
  const text = content.replace(ATTACHED, "").trim();
  return (
    <div className="msg user">
      {ids.length > 0 && (
        <div className="user-attachments">
          {ids.map((id) => (
            <ImageView key={id} api={api} id={id} alt="Attached image" />
          ))}
        </div>
      )}
      {text}
    </div>
  );
}

function ToolCard({
  api,
  item,
  onDecide,
}: {
  api: Api;
  item: Extract<Item, { kind: "tool" }>;
  onDecide: (id: string, approve: boolean, always?: boolean) => void;
}) {
  const media = MEDIA_TOOLS.has(item.name);
  const args = typeof item.args === "object" ? item.args : {};
  const preview =
    item.name === "write_file"
      ? String(args.content ?? "")
      : item.name === "edit_file"
        ? `- ${String(args.old_text ?? "").split("\n").join("\n- ")}\n+ ${String(args.new_text ?? "").split("\n").join("\n+ ")}`
        : "";
  const awaiting = item.status === "awaiting";

  return (
    <div className={`tool-card ${item.status} ${media ? "media-tool" : ""}`}>
      <details open={awaiting}>
        <summary>
          <span className={`tool-status ${item.status}`} aria-label={item.status} />
          <span className="tool-label">{TOOL_LABEL[item.name] ?? item.name}</span>
          {item.summary && <code className="tool-summary">{item.summary}</code>}
          {awaiting && <span className="tool-waiting">needs approval</span>}
          {media && item.status === "running" && <span className="tool-waiting muted">working…</span>}
        </summary>
        {preview && <pre className="tool-preview">{preview.length > 4000 ? `${preview.slice(0, 4000)}\n…` : preview}</pre>}
        {item.output != null && <pre className="tool-output">{item.output}</pre>}
        {typeof item.args === "string" && <pre className="tool-output">{item.args}</pre>}
      </details>
      {item.attachments?.map((a, i) => (
        <div key={i} className="tool-media">
          <AttachmentView api={api} attachment={a} />
        </div>
      ))}
      {awaiting && (
        <div className="tool-actions">
          <button className="primary" onClick={() => onDecide(item.id, true)}>
            Approve
          </button>
          <button className="secondary" onClick={() => onDecide(item.id, true, true)}>
            Approve all in this run
          </button>
          <button className="link danger" onClick={() => onDecide(item.id, false)}>
            Deny
          </button>
        </div>
      )}
    </div>
  );
}

function WorkspacePicker({
  api,
  value,
  workspaces,
  disabled,
  onChange,
  onCreated,
}: {
  api: Api;
  value: string;
  workspaces: WorkspaceInfo[] | null;
  disabled: boolean;
  onChange: (name: string) => void;
  onCreated: (name: string) => Promise<void>;
}) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [gitUrl, setGitUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function create() {
    setBusy(true);
    setError("");
    try {
      const derived = name.trim() || gitUrl.trim().split("/").pop()?.replace(/\.git$/, "") || "";
      await api.createWorkspace(derived, gitUrl.trim() || undefined);
      await onCreated(derived);
      setCreating(false);
      setName("");
      setGitUrl("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="workspace-picker">
      <select
        value={workspaces?.some((w) => w.name === value) ? value : ""}
        disabled={disabled}
        onChange={(e) => (e.target.value === "__new" ? setCreating(true) : onChange(e.target.value))}
        aria-label="Workspace"
      >
        <option value="" disabled>
          Workspace…
        </option>
        {workspaces?.map((w) => (
          <option key={w.name} value={w.name}>
            {w.name}
            {w.git?.branch ? ` (${w.git.branch}${w.git.changes ? `, ${w.git.changes} changed` : ""})` : ""}
          </option>
        ))}
        <option value="__new">+ New workspace…</option>
      </select>
      {creating && (
        <div className="workspace-form card" onKeyDown={(e) => e.key === "Escape" && setCreating(false)}>
          <strong>New workspace</strong>
          <input
            autoFocus
            placeholder="Git URL to clone (optional), e.g. https://github.com/you/repo"
            value={gitUrl}
            onChange={(e) => setGitUrl(e.target.value)}
          />
          <input placeholder="Name (defaults to the repo name)" value={name} onChange={(e) => setName(e.target.value)} />
          {error && <p className="error small">{error}</p>}
          <div className="row-actions start">
            <button className="primary" disabled={busy || (!name.trim() && !gitUrl.trim())} onClick={create}>
              {busy ? (gitUrl ? "Cloning…" : "Creating…") : "Create"}
            </button>
            <button className="link" onClick={() => setCreating(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </span>
  );
}

function SpeakButton({ api, model, text }: { api: Api; model: string; text: string }) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const [error, setError] = useState("");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => () => audio.current?.pause(), []);

  async function toggle() {
    if (state === "playing") {
      audio.current?.pause();
      setState("idle");
      return;
    }
    setState("loading");
    setError("");
    try {
      const url = URL.createObjectURL(await api.speak(model, text));
      const a = new Audio(url);
      routeAudio(a);
      a.onended = () => {
        setState("idle");
        URL.revokeObjectURL(url);
      };
      audio.current = a;
      await a.play();
      setState("playing");
    } catch (e) {
      setState("idle");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <>
    <button className="icon-button" onClick={toggle} title="Read aloud" disabled={state === "loading"}>
      {state === "loading" ? (
        "…"
      ) : state === "playing" ? (
        <>
          <Icon name="stop" size={14} /> Stop
        </>
      ) : (
        <>
          <Icon name="volume" size={14} /> Read aloud
        </>
      )}
    </button>
    {error && <span className="error small speak-error">Couldn't read this aloud: {error}</span>}
    </>
  );
}

function MicButton({
  api,
  model,
  onText,
  onError,
}: {
  api: Api;
  model?: string;
  onText: (text: string) => void;
  onError: (msg: string) => void;
}) {
  const { recording, start, stop } = useRecorder();
  const [transcribing, setTranscribing] = useState(false);
  const available = canRecord();

  async function toggle() {
    if (!model) {
      onError("Dictation needs a speech-to-text model. Install Whisper from Models (Transcription), then the mic works here.");
      return;
    }
    if (!available) {
      onError(
        "Browsers only allow the microphone on secure pages (HTTPS or localhost), and this page is plain HTTP. " +
          "Open Amy through HTTPS (a reverse proxy or Tailscale), or in Chrome allow this address at " +
          "chrome://flags/#unsafely-treat-insecure-origin-as-secure. You can also attach a recording with the paperclip.",
      );
      return;
    }
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
      disabled={transcribing}
      title={
        !model
          ? "Dictate (install a speech-to-text model first)"
          : available
            ? recording
              ? "Stop and transcribe"
              : "Dictate"
            : "Microphone access needs HTTPS (or localhost)"
      }
      onClick={toggle}
    >
      {transcribing ? (
        "…"
      ) : recording ? (
        <>
          <Icon name="stop" size={14} /> Stop
        </>
      ) : (
        <Icon name="mic" size={18} label="Dictate" />
      )}
    </button>
  );
}
