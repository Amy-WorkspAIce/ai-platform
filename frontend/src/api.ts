// Thin client for the platform's OpenAI-compatible API.

const KEY_STORAGE = "amy.apiKey";

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    localStorage.setItem(KEY_STORAGE, key);
  } catch {
    // Storage unavailable (private mode); key lives only for this session.
  }
}

export class AuthError extends Error {}

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

function headers(key: string): HeadersInit {
  return { Authorization: `Bearer ${key}`, "Content-Type": "application/json" };
}

export async function listModels(key: string): Promise<string[]> {
  const resp = await fetch("/v1/models", { headers: headers(key) });
  if (resp.status === 401) throw new AuthError("Invalid API key");
  if (!resp.ok) throw new Error(`Failed to list models (${resp.status})`);
  const body: { data: { id: string }[] } = await resp.json();
  return body.data.map((m) => m.id);
}

/** Stream a chat completion, calling onDelta with each text chunk. */
export async function streamChat(
  key: string,
  model: string,
  messages: ChatMessage[],
  onDelta: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: headers(key),
    body: JSON.stringify({ model, messages, stream: true }),
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
      const data = line.replace(/^data: ?/, "").trim();
      if (!line.startsWith("data:") || !data || data === "[DONE]") continue;
      const delta = JSON.parse(data).choices?.[0]?.delta?.content;
      if (delta) onDelta(delta);
    }
  }
}
