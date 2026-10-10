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
  deleteConversation = (id: string) =>
    this.request<void>(`/api/conversations/${id}`, { method: "DELETE" });
  addMessage = (id: string, message: ChatMessage & { model?: string }) =>
    this.request<unknown>(`/api/conversations/${id}/messages`, {
      method: "POST",
      body: JSON.stringify(message),
    });

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
  pull = (model: string) =>
    this.request<PullState>("/api/models/pull", { method: "POST", body: JSON.stringify({ model }) });
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

export type Capability = "chat" | "image" | "transcription" | "speech";

export type ModelInfo = {
  id: string;
  capability: Capability;
  name?: string;
  default_size?: number;
  negative_prompt?: boolean;
};

export type Conversation = {
  id: string;
  title: string;
  model: string | null;
  created_at: number;
  updated_at: number;
};

export type ConversationDetail = Conversation & {
  messages: (ChatMessage & { id: string; reasoning: string | null })[];
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
  size_gb: number;
  needed_gb: number;
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
