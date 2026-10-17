export type AgentModelProvider = 'api' | 'local';

export class AuthError extends Error {
  constructor(message = '请先登录') {
    super(message);
    this.name = 'AuthError';
  }
}

export interface AuthUser {
  authenticated: boolean;
  username: string;
}

export interface AgentChatResponse {
  conversation_id: string;
  answer: string;
  tool_called: boolean;
  tool_result: null | {
    obj_id?: number;
    final_group?: string;
    glb_url?: string;
    obj_url?: string;
    model_name?: string;
    dynasty?: string;
    confidence?: number;
    top5?: Array<{group: string; score: number}>;
  };
  rag_hits: Array<{chunk_id: string; source: string; page?: number; score?: number; text?: string}>;
  render_instruction: null | {
    type: string;
    model_id?: number;
    model_url?: string;
    model_format?: 'glb' | 'obj';
    confidence?: number;
  };
  tool_calls: Array<{
    tool: string;
    success: boolean;
    message?: string | null;
    result?: Record<string, unknown> | null;
    error?: string | null;
  }>;
  artifacts: Array<{
    id: string;
    type: 'image' | 'model' | 'json';
    label: string;
    url?: string | null;
    preview_url?: string | null;
    download_url?: string | null;
    mime_type?: string | null;
    source_tool: string;
  }>;
  ui_instruction?: null | {
    target_view: 'recognition' | 'model' | 'knowledge';
    active_artifact_id?: string | null;
    message?: string | null;
  };
  status_trace: Array<{event: string; message: string}>;
}

export interface AgentHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (response.status === 401) {
    let message = '请先登录';
    try {
      message = await response.text() || message;
    } catch {
      // ignore read errors
    }
    throw new AuthError(message);
  }
  if (!response.ok) {
    throw new Error((await response.text()) || `请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const response = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ username, password }),
  });
  return parseJsonResponse<AuthUser>(response);
}

export async function logout(): Promise<void> {
  const response = await fetch('/auth/logout', {
    method: 'POST',
    credentials: 'include',
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `退出失败：${response.status}`);
  }
}

export async function getCurrentUser(): Promise<AuthUser> {
  const response = await fetch('/auth/me', {
    credentials: 'include',
  });
  return parseJsonResponse<AuthUser>(response);
}

export async function sendAgentChat(
  message: string,
  conversationId: string,
  history: AgentHistoryMessage[],
  modelProvider: AgentModelProvider = 'api',
  image?: File | File[],
): Promise<AgentChatResponse> {
  const form = new FormData();
  form.append('message', message);
  form.append('conversation_id', conversationId);
  form.append('history', JSON.stringify(history));
  form.append('model_provider', modelProvider);
  if (Array.isArray(image)) {
    image.forEach((item) => form.append('images', item));
  } else if (image) {
    form.append('image', image);
  }

  const response = await fetch('/agent/chat', {
    method: 'POST',
    body: form,
    credentials: 'include',
  });

  return parseJsonResponse<AgentChatResponse>(response);
}

export interface AgentStreamHandlers {
  onAnswerDelta?: (content: string) => void;
}

function buildAgentChatForm(
  message: string,
  conversationId: string,
  history: AgentHistoryMessage[],
  modelProvider: AgentModelProvider,
  images?: File[],
) {
  const form = new FormData();
  form.append('message', message);
  form.append('conversation_id', conversationId);
  form.append('history', JSON.stringify(history));
  form.append('model_provider', modelProvider);
  if (images?.length) {
    images.forEach((image) => form.append('images', image));
  }
  return form;
}

function parseSseChunk(buffer: string) {
  const separator = buffer.includes('\r\n\r\n') ? '\r\n\r\n' : '\n\n';
  const parts = buffer.split(separator);
  return {
    events: parts.slice(0, -1),
    rest: parts[parts.length - 1] ?? '',
  };
}

function readSseEvent(rawEvent: string) {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of rawEvent.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  return { event, data: dataLines.join('\n') };
}

export async function streamAgentChat(
  message: string,
  conversationId: string,
  history: AgentHistoryMessage[],
  modelProvider: AgentModelProvider,
  image: File | File[] | undefined,
  handlers: AgentStreamHandlers = {},
): Promise<AgentChatResponse> {
  const images = Array.isArray(image) ? image : image ? [image] : undefined;
  const response = await fetch('/agent/stream', {
    method: 'POST',
    body: buildAgentChatForm(message, conversationId, history, modelProvider, images),
    credentials: 'include',
  });

  if (response.status === 401) {
    throw new AuthError((await response.text()) || '请先登录');
  }

  if (!response.ok || !response.body) {
    throw new Error((await response.text()) || `Agent 请求失败：${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResponse: AgentChatResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.rest;

    for (const rawEvent of parsed.events) {
      const { event, data } = readSseEvent(rawEvent);
      if (!data || data === '[DONE]') {
        continue;
      }
      if (event === 'answer_delta') {
        const payload = JSON.parse(data) as { content?: string };
        if (payload.content) {
          handlers.onAnswerDelta?.(payload.content);
        }
      } else if (event === 'final') {
        finalResponse = JSON.parse(data) as AgentChatResponse;
      } else if (event === 'error') {
        const payload = JSON.parse(data) as { error?: string };
        throw new Error(payload.error || 'Agent 流式请求失败');
      }
    }

    if (done) {
      break;
    }
  }

  if (!finalResponse) {
    throw new Error('Agent 流式请求没有返回最终结果');
  }
  return finalResponse;
}

export async function synthesizeSpeech(text: string, rate = 1): Promise<Blob> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch('/agent/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      signal: controller.signal,
      body: JSON.stringify({ text, voice: 'boy', rate }),
    });

    if (!response.ok) {
      throw new Error((await response.text()) || `TTS 请求失败：${response.status}`);
    }

    return response.blob();
  } finally {
    window.clearTimeout(timeoutId);
  }
}
