/** What the AI is doing right now, worked out from what the chat has received so far. */

export type ChatStatus = { text: string; detail?: string; opens?: "thinking" | "subagents" };

type StatusItem =
  | { kind: "user" | "notice" }
  | { kind: "assistant"; hasAnswer: boolean }
  | { kind: "tool"; name: string; summary: string; status: string };

type StatusAgent = { status: string; model: string; task: string };

const DOING: Record<string, string> = {
  web_search: "Searching the web",
  fetch_url: "Reading",
  list_files: "Looking through files",
  read_file: "Reading",
  search_files: "Searching files",
  write_file: "Writing",
  edit_file: "Editing",
  run_command: "Running",
  generate_image: "Creating an image",
  edit_image: "Editing the image",
  generate_video: "Creating a video",
  speak: "Recording a voice",
  ask_models: "Asking other models",
  recall_memory: "Checking memory",
};

// Their summary is a long prompt, which says little in a one-line status.
const NO_DETAIL = new Set(["generate_image", "edit_image", "generate_video", "speak", "ask_models"]);

/** "docs_append_text" -> "Docs: append text". */
export const appTool = (name: string) => {
  const [app, ...rest] = name.split("_");
  return rest.length ? `${app.charAt(0).toUpperCase()}${app.slice(1)}: ${rest.join(" ")}` : name;
};

const oneLine = (text: string) => text.replace(/\s+/g, " ").trim();

export function statusOf(items: StatusItem[], agents: StatusAgent[]): ChatStatus {
  const tools = items.filter((i) => i.kind === "tool");
  const awaiting = tools.find((t) => t.status === "awaiting");
  if (awaiting) return { text: "Waiting for your approval", detail: oneLine(awaiting.summary) || undefined };

  const working = agents.filter((a) => a.status === "running");
  if (working.length === 1) {
    return { text: `Sub-agent working (${working[0].model.split("/").pop()})`, detail: oneLine(working[0].task), opens: "subagents" };
  }
  if (working.length > 1) return { text: `${working.length} sub-agents working`, opens: "subagents" };

  const running = tools.filter((t) => t.status === "running");
  if (running.length > 1) return { text: `Running ${running.length} tools` };
  if (running.length === 1) {
    const { name, summary } = running[0];
    return { text: DOING[name] ?? `Using ${appTool(name)}`, detail: NO_DETAIL.has(name) ? undefined : oneLine(summary) || undefined };
  }

  const last = [...items].reverse().find((i) => i.kind !== "notice");
  if (last?.kind === "tool") return { text: "Reading the results" };
  if (last?.kind === "assistant" && last.hasAnswer) return { text: "Writing" };
  return { text: "Thinking", opens: "thinking" };
}
