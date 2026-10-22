// Client for the platform API: OpenAI-compatible /v1 plus management /api.

const KEY_STORAGE = "amy.apiKey";

function storageGet(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function storageSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable (private mode); value lives only for this session.
  }
}

export const getApiKey = () => storageGet(KEY_STORAGE);
export const setApiKey = (key: string) => storageSet(KEY_STORAGE, key);
/** The hub's Open button signs you in by passing the key after "#amy-key=" (the part of a URL
 * browsers never send to the server). Save it and clean it out of the address bar. */
export function takeKeyFromUrl(): void {
  const match = window.location.hash.match(/amy-key=([^&]+)/);
  if (!match) return;
  setApiKey(decodeURIComponent(match[1]));
  history.replaceState(null, "", window.location.pathname + window.location.search);
}

export const getPref = (name: string) => storageGet(`amy.${name}`);
export const setPref = (name: string, value: string) => storageSet(`amy.${name}`, value);

export class AuthError extends Error {}

export class Api {
  constructor(private key: string) {}

  private async raw(path: string, init: RequestInit = {}): Promise<Response> {
    const isForm = init.body instanceof FormData;
    const resp = await fetch(path, {
      ...init,
      headers: {
        Authorization: `Bearer ${this.key}`,
        ...(isForm ? {} : { "Content-Type": "application/json" }),
        ...init.headers,
      },
    });
    if (resp.status === 401) throw new AuthError("Invalid API key");
    if (!resp.ok) {
      let detail = await resp.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return resp;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const resp = await this.raw(path, init);
    return (resp.status === 204 ? undefined : await resp.json()) as T;
  }

  async listModels(): Promise<ModelInfo[]> {
    const body = await this.request<{ data: ModelInfo[] }>("/v1/models");
    return body.data;
  }

  // ---- conversations ----
  conversations = () => this.request<Conversation[]>("/api/conversations");
  conversation = (id: string) => this.request<ConversationDetail>(`/api/conversations/${id}`);
  createConversation = (title: string, model: string) =>
    this.request<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title, model }),
    });
  renameConversation = (id: string, title: string) =>
    this.request<Conversation>(`/api/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
  moveConversation = (id: string, folderId: string | null) =>
    this.request<Conversation>(`/api/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ folder_id: folderId ?? "" }),
    });

  // ---- folders ----
  folders = () => this.request<Folder[]>("/api/folders");
  createFolder = (name: string) =>
    this.request<Folder>("/api/folders", { method: "POST", body: JSON.stringify({ name }) });
  updateFolder = (id: string, patch: Partial<Pick<Folder, "name" | "collapsed" | "position">>) =>
    this.request<Folder>(`/api/folders/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
  deleteFolder = (id: string) => this.request<void>(`/api/folders/${id}`, { method: "DELETE" });

  // ---- memory ----
  memory = (q?: string) => this.request<MemoryState>(`/api/memory${q ? `?q=${encodeURIComponent(q)}` : ""}`);
  addFact = (text: string, category: string) =>
    this.request<Fact>("/api/memory", { method: "POST", body: JSON.stringify({ text, category }) });
  updateFact = (id: string, patch: { text?: string; category?: string }) =>
    this.request<Fact>(`/api/memory/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
  deleteFact = (id: string) => this.request<void>(`/api/memory/${id}`, { method: "DELETE" });
  forgetEverything = () => this.request<{ deleted: number }>("/api/memory?confirm=true", { method: "DELETE" });
  memorySettings = (patch: { enabled?: boolean; auto?: boolean }) =>
    this.request<{ enabled: boolean; auto: boolean }>("/api/memory/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    });

  deleteConversation = (id: string) =>
    this.request<void>(`/api/conversations/${id}`, { method: "DELETE" });
  addMessage = (id: string, message: ChatMessage & { model?: string }) =>
    this.request<unknown>(`/api/conversations/${id}/messages`, {
      method: "POST",
      body: JSON.stringify(message),
    });

  // ---- agent / tools ----
  /** Stream a tool-using chat (web access and/or agent mode) as server-sent events. */
  async agentChat(req: AgentRequest, onEvent: (e: AgentEvent) => void, signal?: AbortSignal) {
    const resp = await this.raw("/api/agent/chat", {
      method: "POST",
      body: JSON.stringify(req),
      signal,
    });
    if (!resp.body) throw new Error("No response body");
    const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (line) onEvent(JSON.parse(line.slice(6)) as AgentEvent);
      }
    }
  }
  approve = (runId: string, callId: string, approve: boolean, always = false) =>
    this.request<{ ok: boolean }>(`/api/agent/runs/${runId}/approve`, {
      method: "POST",
      body: JSON.stringify({ call_id: callId, approve, always }),
    });
  workspaces = () => this.request<WorkspaceInfo[]>("/api/workspaces");
  createWorkspace = (name: string, git_url?: string) =>
    this.request<{ name: string }>("/api/workspaces", {
      method: "POST",
      body: JSON.stringify({ name, git_url: git_url || undefined }),
    });
  deleteWorkspace = (name: string) =>
    this.request<void>(`/api/workspaces/${encodeURIComponent(name)}`, { method: "DELETE" });

  // ---- images ----
  generateImage = (req: ImageRequest) =>
    this.request<{ data: { id: string; b64_json: string }[]; seed: number }>(
      "/v1/images/generations",
      { method: "POST", body: JSON.stringify(req) },
    );
  images = () => this.request<GalleryImage[]>("/api/images");
  deleteImage = (id: string) => this.request<void>(`/api/images/${id}`, { method: "DELETE" });
  async imageUrl(id: string): Promise<string> {
    const blob = await (await this.raw(`/api/images/${id}`)).blob();
    return URL.createObjectURL(blob);
  }

  // ---- audio ----
  async transcribe(model: string, audio: Blob, filename: string): Promise<string> {
    const form = new FormData();
    form.append("model", model);
    form.append("file", audio, filename);
    const body = await this.request<{ text: string }>("/v1/audio/transcriptions", {
      method: "POST",
      body: form,
    });
    return body.text;
  }
  async speak(model: string, input: string, voice?: string, speed = 1): Promise<Blob> {
    const resp = await this.raw("/v1/audio/speech", {
      method: "POST",
      body: JSON.stringify({ model, input, voice, speed, response_format: "wav" }),
    });
    return resp.blob();
  }
  voices = (model: string) =>
    this.request<{ voices: string[] }>(`/api/audio/voices?model=${encodeURIComponent(model)}`);

  catalog = () => this.request<Catalog>("/api/catalog");
  installed = () => this.request<InstalledModel[]>("/api/models/installed");
  pulls = () => this.request<PullState[]>("/api/models/pulls");
  pull = (model: string, engine?: string) =>
    this.request<PullState>("/api/models/pull", {
      method: "POST",
      body: JSON.stringify({ model, engine }),
    });
  /** The whole model table (filtered and sorted in the browser). */
  catalogModels = () => this.request<ModelTable>("/api/catalog/models");
  /** Search Hugging Face beyond the standard listing. */
  searchModels = (q: string) =>
    this.request<{ models: ModelRow[] }>(`/api/catalog/search?q=${encodeURIComponent(q)}`);

  // ---- settings ----
  settings = () => this.request<AppSettings>("/api/settings");
  saveHfToken = (token: string) =>
    this.request<HfStatus>("/api/settings/huggingface", {
      method: "PUT",
      body: JSON.stringify({ token }),
    });
  clearHfToken = () => this.request<HfStatus>("/api/settings/huggingface", { method: "DELETE" });

  async uploadImage(file: File): Promise<string> {
    const form = new FormData();
    form.append("file", file, file.name);
    return (await this.request<{ id: string }>("/api/images/upload", { method: "POST", body: form })).id;
  }
  saveModelDefault = (capability: string, model: string | null) =>
    this.request<AppSettings["models"]>("/api/settings/models", {
      method: "PUT",
      body: JSON.stringify({ capability, model }),
    });
  saveWebAccess = (enabled: boolean) =>
    this.request<{ web_access: boolean }>("/api/settings/web", {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    });

  // ---- audio library ----
  audioList = () => this.request<GalleryAudio[]>("/api/audio");
  deleteAudio = (id: string) => this.request<void>(`/api/audio/${id}`, { method: "DELETE" });
  async audioUrl(id: string): Promise<string> {
    return URL.createObjectURL(await (await this.raw(`/api/audio/${id}`)).blob());
  }

  // ---- video ----
  generateVideo = (req: VideoRequest) =>
    this.request<{ id: string }>("/v1/videos/generations", { method: "POST", body: JSON.stringify(req) });
  videoJobs = () => this.request<VideoJob[]>("/api/videos/jobs");
  videos = () => this.request<GalleryVideo[]>("/api/videos");
  deleteVideo = (id: string) => this.request<void>(`/api/videos/${id}`, { method: "DELETE" });
  async videoUrl(id: string): Promise<string> {
    return URL.createObjectURL(await (await this.raw(`/api/videos/${id}`)).blob());
  }
  deleteModel = (model: string) =>
    this.request<void>(`/api/models/${model.split("/").map(encodeURIComponent).join("/")}`, {
      method: "DELETE",
    });

  /** Stream a chat completion. Calls onDelta with content and/or reasoning ("thinking") text. */
  async streamChat(
    model: string,
    messages: ChatMessage[],
    onDelta: (d: { content?: string; reasoning?: string }) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${this.key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: messages.map(({ role, content }) => ({ role, content })),
        stream: true,
      }),
      signal,
    });
    if (resp.status === 401) throw new AuthError("Invalid API key");
    if (!resp.ok || !resp.body) throw new Error(`Chat failed (${resp.status}): ${await resp.text()}`);

    const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        const delta = JSON.parse(data).choices?.[0]?.delta ?? {};
        const reasoning = delta.reasoning ?? delta.reasoning_content;
        if (delta.content || reasoning) onDelta({ content: delta.content, reasoning });
      }
    }
  }
}

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
  reasoning?: string;
};

export type Fit = "gpu" | "light" | "partial" | "cpu" | "too_large" | "unknown";

export type SortKey = "perfect" | "fit" | "size_asc" | "size_desc" | "name" | "popular";

/** One installable model: a variant plus its family's details. */
export type ModelRow = Variant & {
  family_id: string;
  name: string;
  publisher: string;
  license: string | null;
  description: string;
  engine: string;
  source?: "curated" | "ollama" | "huggingface";
  url?: string;
  tools?: boolean;
  popularity?: number;
  downloads?: number;
  downloads_scope?: "model" | "family";
  /** When the model was last updated (epoch seconds), if the source says. */
  updated?: number | null;
  unsupported?: string | null;
  engine_note: string | null;
  engine_installed: boolean;
};

export type ModelTable = { models: ModelRow[]; hardware: Hardware; hf_token_set: boolean };

export type HfStatus = { token_set: boolean; username: string | null; source: "env" | "ui" | null };

export type ModelChoice = {
  pinned: string | null;
  best: string | null;
  installed: { id: string; name: string; fit: Fit }[];
};

export type AppSettings = {
  version: string;
  huggingface: HfStatus;
  search: { providers: string[]; bundled_searxng: boolean };
  web_access: boolean;
  models: Record<"image" | "video" | "speech" | "transcription", ModelChoice>;
};

export type GalleryAudio = {
  id: string;
  model: string;
  text: string;
  created_at: number;
  params: { voice: string; speed: number };
};

export type Attachment =
  | { type: "image"; id: string }
  | { type: "video"; job_id: string }
  | { type: "audio"; id: string };

export type VideoRequest = {
  model: string;
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  frames?: number;
  fps?: number;
  steps?: number;
  seed?: number;
};

export type VideoJob = {
  id: string;
  model: string;
  prompt: string;
  status: "queued" | "running" | "done" | "error";
  phase: string;
  step: number;
  steps: number;
  error: string | null;
  started_at: number;
  finished_at?: number;
  params: { width: number; height: number; frames: number; fps: number };
};

export type GalleryVideo = {
  id: string;
  model: string;
  prompt: string;
  created_at: number;
  params: {
    width: number;
    height: number;
    frames: number;
    fps: number;
    seed: number;
    steps: number | null;
    seconds: number | null;
    negative_prompt: string | null;
  };
};

export type Capability = "chat" | "image" | "video" | "transcription" | "speech";

export type ModelInfo = {
  id: string;
  capability: Capability;
  tools?: boolean | null;
  name?: string;
  default_size?: number;
  negative_prompt?: boolean;
  default_width?: number;
  default_height?: number;
  default_frames?: number;
  fps?: number;
};

export type ChatSettings = { mode?: "chat" | "agent"; web?: boolean; workspace?: string | null };

export type Folder = { id: string; name: string; position: number; collapsed: boolean; created_at: number };

export type Fact = {
  id: string;
  text: string;
  category: string;
  source_conversation_id: string | null;
  created_at: number;
  updated_at: number;
  last_used_at: number | null;
};

export type MemoryState = {
  facts: Fact[];
  enabled: boolean;
  auto: boolean;
  embedding_model: string | null;
  categories: string[];
};

export type Conversation = {
  id: string;
  title: string;
  model: string | null;
  folder_id?: string | null;
  settings: ChatSettings;
  created_at: number;
  updated_at: number;
};

export type StoredMessage = {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning: string | null;
  meta: {
    tool_calls?: { id: string; function: { name: string; arguments: string } }[];
    tool_call_id?: string;
    name?: string;
    ok?: boolean;
    attachments?: Attachment[];
  } | null;
};

export type ConversationDetail = Conversation & { messages: StoredMessage[] };

export type AgentRequest = {
  model: string;
  message: string;
  conversation_id?: string | null;
  mode: "chat" | "agent";
  workspace?: string | null;
  auto_approve: boolean;
  folder_id?: string | null;
};

export type AgentEvent =
  | { type: "start"; run_id: string; conversation_id: string }
  | { type: "text" | "reasoning"; delta: string }
  | {
      type: "tool_call";
      id: string;
      name: string;
      arguments: Record<string, unknown> | string;
      summary: string;
      needs_approval: boolean;
    }
  | { type: "tool_result"; id: string; ok: boolean; output: string; attachments?: Attachment[] }
  | { type: "notice"; message: string }
  | { type: "answer_done" }
  | { type: "memory"; action: "remembered" | "updated" | "removed"; text: string }
  | { type: "error"; message: string }
  | { type: "done" };

export type WorkspaceInfo = {
  name: string;
  modified: number;
  git: { branch: string | null; remote: string | null; changes: number } | null;
};

export type ImageRequest = {
  model: string;
  prompt: string;
  size?: string;
  n?: number;
  negative_prompt?: string;
  steps?: number;
  seed?: number;
};

export type GalleryImage = {
  id: string;
  model: string;
  prompt: string;
  created_at: number;
  params: {
    width: number;
    height: number;
    steps: number;
    guidance: number;
    seed: number;
    negative_prompt: string | null;
    seconds: number | null;
  };
};

export type Variant = {
  tag: string;
  params: string;
  size_gb: number | null;
  size_estimated?: boolean;
  requires_token?: boolean;
  gated?: boolean;
  best?: boolean;
  needed_gb: number | null;
  use_cases: string[];
  fit: Fit;
  installed: boolean;
  disk_ok: boolean;
};

export type Family = {
  id: string;
  name: string;
  publisher: string;
  license: string;
  description: string;
  engine: string;
  engine_available: boolean;
  engine_installed: boolean;
  tools?: boolean;
  source?: "curated" | "ollama" | "huggingface";
  url?: string;
  popularity?: number;
  unsupported?: string | null;
  engine_note: string | null;
  use_cases: string[];
  variants: Variant[];
};

export type Hardware = {
  gpus: { name: string; vram_total_gb: number; vram_free_gb: number }[];
  vram_total_gb: number;
  ram_total_gb: number | null;
  ram_available_gb: number | null;
  cpu_count: number | null;
  disk_free_gb: number | null;
  vram_override: boolean;
};

export type Catalog = {
  hardware: Hardware;
  use_cases: { id: string; name: string; description: string }[];
  engines: { id: string; name: string; available: boolean; note?: string }[];
  families: Family[];
  recommendations: Record<string, string[]>;
  ollama_online: boolean;
  hf_token_set: boolean;
};

export type InstalledModel = {
  name: string;
  size_gb: number;
  family: string | null;
  parameters: string | null;
  quantization: string | null;
};

export type PullState = {
  model: string;
  status: string;
  completed: number;
  total: number;
  done: boolean;
  error: string | null;
};
