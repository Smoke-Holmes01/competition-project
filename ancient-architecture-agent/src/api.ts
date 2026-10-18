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
  rag_hits: Array<{chunk_id: string; source: string; page?: number | null; score?: number; text?: string}>;
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


export interface ProgressEvent { event: string; message: string; tool?: string | null }
export interface Conversation {
  conversation_id: string;
  messages: Array<{role: 'user' | 'assistant'; content: string; time: string; status: string;
    assets: Array<{id: string; filename: string}>}>;
  last_result: AgentChatResponse | null;
}

async function checked(response: Response): Promise<Response> {
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try { const body = await response.json(); message = body.detail || message; } catch { /* no JSON */ }
    throw new Error(message);
  }
  return response;
}

export async function getConversation(id: string, signal?: AbortSignal): Promise<Conversation | null> {
  const response = await fetch(`/agent/conversations/${encodeURIComponent(id)}`, { signal });
  if (response.status === 404) return null;
  return (await checked(response)).json();
}

export async function deleteConversation(id: string): Promise<void> {
  await checked(await fetch(`/agent/conversations/${encodeURIComponent(id)}`, {method: 'DELETE'}));
}

export async function streamAgentChat(
  message: string, conversationId: string, images: File[] | undefined,
  handlers: {onAnswerDelta?: (text: string) => void; onProgress?: (event: ProgressEvent) => void},
  signal?: AbortSignal,
): Promise<AgentChatResponse> {
  const form = new FormData();
  form.append('message', message);
  form.append('conversation_id', conversationId);
  images?.forEach(image => form.append('images', image));
  const response = await checked(await fetch('/agent/stream', {method: 'POST', body: form, signal}));
  if (!response.body) throw new Error('浏览器未提供流式响应。');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: AgentChatResponse | null = null;
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? '';
      for (const block of blocks) {
        const lines = block.split(/\r?\n/);
        const event = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
        const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (!data || data === '[DONE]') continue;
        const payload = JSON.parse(data);
        if (event === 'answer_delta') handlers.onAnswerDelta?.(payload.content || '');
        else if (event === 'final') result = payload;
        else if (event === 'error') throw new Error(payload.error || 'Agent 请求失败');
        else if (payload.message) handlers.onProgress?.(payload);
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  if (!result) throw new Error('请求中断，未收到最终结果。');
  return result;
}
