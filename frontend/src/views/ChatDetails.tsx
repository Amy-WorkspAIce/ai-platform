import { useEffect, useRef, useState } from "react";
import type { ChatDetails as Details, MemoryLogEntry, SubAgent } from "../api";
import { Icon } from "../icons";

/** One user message and what Amy thought and did while answering it. */
export type Turn = {
  prompt: string;
  steps: ({ kind: "thought"; text: string } | { kind: "tool"; label: string; summary: string; ok: boolean | null })[];
};

export type Section = "memory" | "thinking" | "subagents";

/** Which section to bring into view (and which turn, for Thinking); ``nonce`` repeats a request. */
export type Focus = { section: Section; turn?: number; nonce: number };

type Props = {
  details: Details | null;
  turns: Turn[];
  thinkingNow: boolean;
  focus: Focus | null;
  onClose: () => void;
  onForget: (memoryId: string) => void;
  onOpenMemory?: () => void;
};

const LEARNED: Record<string, string> = {
  remembered: "Remembered",
  updated: "Updated",
  removed: "Forgot",
  failed: "Couldn't update",
};

const KIND: Record<SubAgent["kind"], string> = {
  ask: "Asked for its view",
  image: "Image",
  video: "Video",
  speech: "Voice",
};

const STATUS: Record<SubAgent["status"], string> = {
  running: "running",
  done: "ok",
  failed: "error",
  stopped: "error",
};

function duration(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  const m = Math.floor(seconds / 60);
  return m < 60 ? `${m}m ${Math.round(seconds % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Re-renders every second while ``active`` (for running timers). */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

/**
 * The chat's Details panel: things you don't need to follow, kept in one place for when you
 * want to. What memory did in this chat, Amy's thought process, and the other models it
 * handed work to.
 */
export function ChatDetails({ details, turns, thinkingNow, focus, onClose, onForget, onOpenMemory }: Props) {
  const sections = useRef<Record<Section, HTMLDetailsElement | null>>({ memory: null, thinking: null, subagents: null });
  const body = useRef<HTMLDivElement>(null);
  const memories = details?.memories ?? [];
  const learned = memories.filter((m) => m.action !== "recalled");
  const recalled = memories.filter((m) => m.action === "recalled");
  const agents = details?.subagents ?? [];
  const running = agents.filter((a) => a.status === "running").length;
  const thoughts = turns.filter((t) => t.steps.length > 0);

  useEffect(() => {
    if (!focus) return;
    const el = sections.current[focus.section];
    if (!el) return;
    el.open = true;
    const target = (focus.turn != null && body.current?.querySelector(`#details-turn-${focus.turn}`)) || el;
    target.scrollIntoView({ block: "start", behavior: "smooth" });
    if (target instanceof HTMLDetailsElement) target.open = true;
  }, [focus?.nonce]);

  return (
    <aside className="chat-details" aria-label="Chat details">
      <div className="chat-details-head">
        <strong>Details</strong>
        <button type="button" className="icon-button" onClick={onClose} title="Hide details">
          <Icon name="close" size={16} label="Hide details" />
        </button>
      </div>
      <div className="chat-details-body" ref={body}>
        <p className="muted small details-intro">Nothing here needs your attention. It's what Amy did behind the scenes.</p>

        <details className="details-section" open ref={(el) => {
            sections.current.memory = el;
          }}>
          <summary>
            <Icon name="orbit" /> Memory <span className="count">{learned.length + recalled.length}</span>
          </summary>
          <h4 className="label">Learned in this chat</h4>
          {learned.length ? (
            <ul className="details-list">
              {learned.map((m) => (
                <MemoryRow key={m.id} entry={m} onForget={onForget} />
              ))}
            </ul>
          ) : (
            <p className="muted small">Nothing yet. Facts you share about yourself show up here once Amy remembers them.</p>
          )}
          <h4 className="label">Recalled from earlier chats</h4>
          {recalled.length ? (
            <ul className="details-list">
              {recalled.map((m) => (
                <MemoryRow key={m.id} entry={m} onForget={onForget} />
              ))}
            </ul>
          ) : (
            <p className="muted small">None used in this chat.</p>
          )}
          {onOpenMemory && (
            <button type="button" className="link small" onClick={onOpenMemory}>
              Everything Amy remembers
            </button>
          )}
        </details>

        <details className="details-section" open ref={(el) => {
            sections.current.thinking = el;
          }}>
          <summary>
            Thought process <span className="count">{thoughts.length}</span>
            {thinkingNow && <span className="tool-status running" aria-label="thinking" />}
          </summary>
          {thoughts.length ? (
            thoughts.map((t) => {
              const index = turns.indexOf(t);
              return (
                <details key={index} id={`details-turn-${index}`} className="details-turn" open={index === turns.length - 1}>
                  <summary>{t.prompt || "(image)"}</summary>
                  <ol className="details-steps">
                    {t.steps.map((s, i) =>
                      s.kind === "thought" ? (
                        <li key={i} className="details-thought">
                          {s.text}
                        </li>
                      ) : (
                        <li key={i} className="details-tool small">
                          <span className={`tool-status ${s.ok == null ? "running" : s.ok ? "ok" : "error"}`} />
                          <span className="details-tool-label">{s.label}</span>
                          {s.summary && <code>{s.summary}</code>}
                        </li>
                      ),
                    )}
                  </ol>
                </details>
              );
            })
          ) : (
            <p className="muted small">
              No thinking shown yet. Reasoning models (like Qwen 3 or DeepSeek R1) think before answering; their
              thoughts and the tools they use appear here.
            </p>
          )}
        </details>

        <details className="details-section" open ref={(el) => {
            sections.current.subagents = el;
          }}>
          <summary>
            Sub-agents <span className="count">{agents.length}</span>
            {running > 0 && <span className="tool-status running" aria-label={`${running} running`} />}
          </summary>
          {agents.length ? (
            <ul className="details-list">
              {agents.map((a) => (
                <AgentRow key={a.id} agent={a} />
              ))}
            </ul>
          ) : (
            <p className="muted small">
              Other models Amy hands work to show up here: image, video and voice models, and models asked for their
              view when a rule calls for it.
            </p>
          )}
        </details>
      </div>
    </aside>
  );
}

function MemoryRow({ entry, onForget }: { entry: MemoryLogEntry; onForget: (id: string) => void }) {
  const gone = !entry.exists && entry.action !== "failed";
  return (
    <li className={`details-memory ${gone ? "gone" : ""}`}>
      <div>
        {entry.action !== "recalled" && <span className={`tag memory-${entry.action}`}>{LEARNED[entry.action]}</span>}{" "}
        {entry.text}
        {gone && entry.action !== "removed" && <span className="muted small"> (since forgotten)</span>}
      </div>
      {entry.exists && entry.memory_id && entry.action !== "removed" && (
        <button type="button" className="link small danger" onClick={() => onForget(entry.memory_id!)} title="Delete this fact from memory">
          Forget
        </button>
      )}
    </li>
  );
}

function AgentRow({ agent }: { agent: SubAgent }) {
  const now = useNow(agent.status === "running");
  const took = (agent.finished_at ?? now) - agent.started_at;
  return (
    <li className="details-agent">
      <details>
        <summary>
          <span className={`tool-status ${STATUS[agent.status]}`} aria-label={agent.status} />
          <span className="details-agent-model">{agent.model.split("/").pop()}</span>
          <span className="muted small">
            {KIND[agent.kind]} · {agent.status === "running" ? `running ${duration(took)}` : agent.status === "done" ? duration(took) : agent.status}
          </span>
        </summary>
        <p className="details-agent-task small">{agent.task}</p>
        {agent.result && <pre className="tool-output">{agent.result}</pre>}
      </details>
    </li>
  );
}
