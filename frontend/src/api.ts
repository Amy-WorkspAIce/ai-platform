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

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const resp = await fetch(path, {
      ...init,
      headers: {
        Authorization: `Bearer ${this.key}`,
        "Content-Type": "application/json",
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
      throw new Error(`${resp.status}: ${detail}`);
    }
    return (resp.status === 204 ? undefined : await resp.json()) as T;
  }

  async listModels(): Promise<string[]> {
    const body = await this.request<{ data: { id: string }[] }>("/v1/models");
    return body.data.map((m) => m.id);
  }

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

export type Fit = "gpu" | "partial" | "cpu" | "too_large" | "unknown";

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
