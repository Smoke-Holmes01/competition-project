import json

import httpx
from langchain_core.messages import HumanMessage

from app.providers import DeepSeekModel, with_images


async def test_deepseek_real_integration_serialization_async_stream_and_tool_calls(config, png):
    config.deepseek_api_key = "test-only"
    requests = []

    async def handler(request):
        body = json.loads(await request.aread())
        requests.append(body)
        if body.get("stream"):
            chunks = (
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tool1",
                                "type": "function",
                                "function": {"name": "preview_model", "arguments": '{"model_'},
                            }
                        ]
                    },
                    {"tool_calls": [{"index": 0, "function": {"arguments": 'id":1}'}}]},
                ]
                if body.get("tools")
                else [{"content": "答"}, {"content": "案"}]
            )
            data = ""
            for delta in chunks:
                payload = {
                    "id": "chat-test",
                    "object": "chat.completion.chunk",
                    "model": "deepseek-flash",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                data += f"data: {json.dumps(payload)}\n\n"
            data += "data: [DONE]\n\n"
            return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "intent",
                                    "type": "function",
                                    "function": {
                                        "name": "Intent",
                                        "arguments": json.dumps({"kind": "image"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        model = DeepSeekModel(config, http_async_client=http_client)
        intent = await model.classify([HumanMessage(content="看图")], 1)
        assert intent.kind == "image"
        original = HumanMessage(content="看图")
        multimodal = with_images(original, [("a" * 32, png, "image/png")])
        assert original.content == "看图"
        assert "".join([piece async for piece in model.answer([multimodal])]) == "答案"
        tool_schema = {
            "type": "function",
            "function": {
                "name": "preview_model",
                "parameters": {
                    "type": "object",
                    "properties": {"model_id": {"type": "integer"}},
                    "required": ["model_id"],
                },
            },
        }
        merged = None
        async for chunk in model.client.bind_tools([tool_schema]).astream([multimodal]):
            merged = chunk if merged is None else merged + chunk
        assert merged.tool_calls[0]["args"] == {"model_id": 1}
        assert merged.tool_calls[0]["id"] == "tool1"
        assert all(request["thinking"] == {"type": "disabled"} for request in requests)
        assert all(request["model"] == "deepseek-flash" for request in requests)
        assert requests[1]["messages"][0]["content"][-1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        await model.aclose()
