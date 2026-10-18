import assert from 'node:assert/strict';
import {test} from 'node:test';
import {deleteConversation, getConversation, streamAgentChat} from '../src/api.ts';

const complete = {conversation_id: 'one', answer: '回答中文', tool_called: false, tool_result: null,
  rag_hits: [], render_instruction: null, tool_calls: [], artifacts: [], status_trace: []};

function fragmentedSse(text) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({start(controller) {
    // Deliberately split CRLF boundaries and UTF-8 characters across transport chunks.
    for (let index = 0; index < bytes.length; index += 3) controller.enqueue(bytes.slice(index, index + 3));
    controller.close();
  }}), {headers: {'Content-Type': 'text/event-stream'}});
}

test('SSE client decodes partial UTF-8 and preserves tool progress and final result', async () => {
  const original = globalThis.fetch;
  const text = [
    ': heartbeat\r\n\r\n',
    'event: session\r\ndata: {"conversation_id":"one"}\r\n\r\n',
    'event: tool_start\r\ndata: {"event":"tool_start","message":"正在匹配","tool":"match_single_image"}\r\n\r\n',
    'event: answer_delta\r\ndata: {"content":"回答中文"}\r\n\r\n',
    `event: final\r\ndata: ${JSON.stringify(complete)}\r\n\r\n`,
    'event: done\r\ndata: [DONE]\r\n\r\n',
  ].join('');
  let sent;
  globalThis.fetch = async (url, options) => { sent = options.body; return fragmentedSse(text); };
  try {
    const deltas = [], progress = [];
    const result = await streamAgentChat('你好', 'one', undefined, {
      onAnswerDelta: text => deltas.push(text), onProgress: event => progress.push(event),
    });
    assert.deepEqual(result, complete);
    assert.equal(deltas.join(''), '回答中文');
    assert.equal(progress[0].tool, 'match_single_image');
    assert.equal(sent.get('conversation_id'), 'one');
    assert.equal(sent.has('history'), false);
    assert.equal(sent.has('model_provider'), false);
  } finally { globalThis.fetch = original; }
});

test('SSE error or missing final response cannot be mistaken for success', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => fragmentedSse('event: error\ndata: {"error":"工具失败"}\n\n');
    await assert.rejects(streamAgentChat('你好', 'one', undefined, {}), /工具失败/);
    globalThis.fetch = async () => fragmentedSse('event: answer_delta\ndata: {"content":"未完成"}\n\n');
    await assert.rejects(streamAgentChat('你好', 'one', undefined, {}), /未收到最终结果/);
  } finally { globalThis.fetch = original; }
});

test('conversation restoration handles absence and clearing uses DELETE', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(null, {status: 404});
    assert.equal(await getConversation('absent'), null);
    const restored = {conversation_id: 'one', messages: [], last_result: complete};
    globalThis.fetch = async () => Response.json(restored);
    assert.deepEqual(await getConversation('one'), restored);
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/agent/conversations/one');
      assert.equal(options.method, 'DELETE');
      return new Response(null, {status: 204});
    };
    await deleteConversation('one');
  } finally { globalThis.fetch = original; }
});
