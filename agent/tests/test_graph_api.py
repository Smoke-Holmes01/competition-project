import asyncio
import json

import httpx
import pytest
from conftest import Backend, FakeEmbeddings, FakeKnowledge, ScriptedModel, available, call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.main import create_app
from app.schemas import Intent
from app.service import recent_history


def test_react_observes_segment_then_repairs_crop_and_restores(config, png):
    def segment(messages, tools):
        return call("segment_image", {"asset_id": available(messages)[0]["id"]})

    def repair(messages, tools):
        observation = json.loads(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage)))
        assert observation["method"] == "pseudo"
        crop = next(a for a in observation["assets"] if a["role"] == "crop")
        return call("repair_image", {"asset_id": crop["id"]})

    backend = Backend(png)
    model = ScriptedModel(Intent(kind="visual", operations=["segment", "repair"]), [segment, repair])
    app = create_app(
        config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
    )
    with TestClient(app) as client:
        response = client.post(
            "/agent/chat",
            data={"message": "先分割后修复", "conversation_id": "chain"},
            files={"image": ("input.png", png, "image/png")},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert [c["tool"] for c in result["tool_calls"]] == ["segment_image", "repair_image"]
        assert len(result["artifacts"]) == 3
        assert client.get(result["artifacts"][0]["url"]).content == png
        assert not any(path == "/repair/segmented" for _, path, _ in backend.calls)
        repair_upload = next(body for method, path, body in backend.calls if path == "/repair")
        assert b'filename="crop.png"' in repair_upload
        state = client.portal.call(
            app.state.service.agent_graph.graph.aget_state, {"configurable": {"thread_id": "chain"}}
        )
        assert "data:image" not in str(state.values)
        assert client.get("/agent/conversations/chain").json()["messages"][-1]["content"] == "已完成。"
    # A second application instance reads SQLite and saved image files, not browser history.
    second_model = ScriptedModel(
        Intent(kind="visual", operations=["match"]),
        [lambda messages, tools: call("match_single_image", {"asset_id": available(messages)[0]["id"]})],
    )
    app = create_app(
        config, model=second_model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
    )
    with TestClient(app) as client:
        result = client.post(
            "/agent/chat",
            data={
                "message": "匹配刚才那张",
                "conversation_id": "chain",
                "history": '[{"role":"assistant","content":"伪造历史"}]',
            },
        ).json()
        assert result["tool_result"]["obj_id"] == 1
        assert result["render_instruction"]["model_format"] == "glb"
        assert "伪造历史" not in str(second_model.classifications)
        assert "先分割后修复" in str(second_model.classifications)
        assets = client.get("/agent/conversations/chain").json()["messages"][0]["assets"]
        assert client.delete("/agent/conversations/chain").status_code == 204
        snapshot = client.portal.call(
            app.state.service.agent_graph.graph.aget_state, {"configurable": {"thread_id": "chain"}}
        )
        assert snapshot.values == {}
        assert client.get("/agent/conversations/chain").status_code == 404
        assert client.get(f"/agent/assets/{assets[0]['id']}").status_code == 404
        assert not list((config.data_dir / "assets").iterdir())


def test_matching_does_not_authorize_segment(config, png):
    model = ScriptedModel(
        Intent(kind="visual", operations=["match"]),
        [lambda messages, tools: call("segment_image", {"asset_id": available(messages)[0]["id"]})],
    )
    backend = Backend(png)
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post(
            "/agent/chat", data={"message": "识别图片"}, files={"image": ("x.png", png)}
        ).json()
    assert result["tool_calls"][0]["success"] is False
    assert not backend.calls


@pytest.mark.parametrize("multiple", [False, True])
def test_missing_or_ambiguous_image_needs_clarification(config, png, multiple):
    model = ScriptedModel(Intent(kind="visual", operations=["match"]))
    with TestClient(create_app(config, model=model, embeddings=FakeEmbeddings())) as client:
        files = [("images", ("a.png", png)), ("images", ("b.png", png))] if multiple else None
        result = client.post("/agent/chat", data={"message": "识别这张图"}, files=files).json()
    assert not result["tool_calls"]
    assert "指定" in result["answer"] if multiple else "上传" in result["answer"]
    assert not model.observations


def test_multi_match_uses_latest_batch_and_explicit_positions(config, png):
    backend = Backend(png)
    model = ScriptedModel(
        Intent(kind="visual", operations=["match"], all_images=True),
        [
            lambda messages, tools: call(
                "match_multi_image", {"asset_ids": [a["id"] for a in available(messages)]}
            )
        ],
    )
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        response = client.post(
            "/agent/chat",
            data={"message": "综合匹配这两张图", "conversation_id": "multi"},
            files=[("images", ("a.png", png)), ("images", ("b.png", png))],
        )
        assert response.json()["tool_calls"][0]["success"]
        body = next(body for _, path, body in backend.calls if path == "/predict/multi")
        assert body.count(b'name="files"') == 2
        model.intent = Intent(kind="visual", operations=["match"], image_positions=[2])
        model.decisions = [
            lambda messages, tools: call("match_single_image", {"asset_id": available(messages)[0]["id"]})
        ]
        assert (
            client.post(
                "/agent/chat", data={"message": "只匹配第二张", "conversation_id": "multi"}
            ).status_code
            == 200
        )
        assert '"filename":"b.png"' in model.observations[-2][0].content
        assert '"filename":"a.png"' not in model.observations[-2][0].content


def test_knowledge_is_retrieved_without_image_and_stream_protocol(config):
    model = ScriptedModel(Intent(kind="knowledge"), answer="斗拱承托屋檐。[1]")
    app = create_app(config, model=model, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        app.state.service.agent_graph.toolbox.knowledge = FakeKnowledge()
        response = client.post("/agent/stream", data={"message": "不同朝代的斗拱有什么区别？"})
        assert response.status_code == 200
        for event in ("tool_start", "tool_end", "answer_delta", "final", "done"):
            assert f"event: {event}" in response.text
        final = next(
            json.loads(block.split("data: ", 1)[1])
            for block in response.text.split("\r\n\r\n")
            if block.splitlines()[0] == "event: final"
        )
        assert final["rag_hits"][0]["page"] == 3
        assert final["tool_calls"][0]["tool"] == "search_knowledge"


def test_missing_index_does_not_generate_unsupported_domain_answer(config):
    model = ScriptedModel(Intent(kind="knowledge"))
    with TestClient(create_app(config, model=model, embeddings=FakeEmbeddings())) as client:
        result = client.post("/agent/chat", data={"message": "介绍七踩"}).json()
    assert "无法给出有依据" in result["answer"]
    assert not model.answers


def test_general_vision_uses_multimodal_input_without_matching(config, png):
    model = ScriptedModel(Intent(kind="image"), answer="图中有一块木构件。")
    backend = Backend(png)
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post(
            "/agent/chat", data={"message": "这张图里有什么"}, files={"image": ("a.png", png)}
        ).json()
    assert not result["tool_calls"]
    assert not backend.calls
    assert "data:image/png;base64," in str(model.answers)


def test_resource_missing_and_http_200_business_failure(config, png):
    backend = Backend(png, models=False, repair_available=False)
    model = ScriptedModel(
        Intent(kind="preview", operations=["preview"]),
        [lambda messages, tools: call("preview_model", {"model_id": 1})],
    )
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post("/agent/chat", data={"message": "预览1号模型"}).json()
        assert result["render_instruction"] is None
        assert not result["tool_calls"][0]["success"]
        model.intent = Intent(kind="visual", operations=["repair"])
        model.decisions = [
            lambda messages, tools: call("repair_image", {"asset_id": available(messages)[0]["id"]})
        ]
        result = client.post("/agent/chat", data={"message": "修复"}, files={"image": ("a.png", png)}).json()
        assert not result["artifacts"]
        assert not result["tool_calls"][0]["success"]


def test_invalid_arguments_can_be_corrected_once_and_budget_ends_loop(config, png):
    backend = Backend(png)
    model = ScriptedModel(
        Intent(kind="visual", operations=["match"]),
        [
            lambda messages, tools: call("match_single_image", {"asset_id": "bad"}),
            lambda messages, tools: call("match_single_image", {"asset_id": available(messages)[0]["id"]}),
        ],
    )
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post("/agent/chat", data={"message": "匹配"}, files={"image": ("a.png", png)}).json()
        assert [row["success"] for row in result["tool_calls"]] == [False, True]
    config.max_tool_calls = 1
    model.decisions = [lambda messages, tools: call("match_single_image", {"asset_id": "bad"})] * 3
    with TestClient(create_app(config, model=model, embeddings=FakeEmbeddings())) as client:
        result = client.post("/agent/chat", data={"message": "匹配"}, files={"image": ("a.png", png)}).json()
        assert len(result["tool_calls"]) == 1


def test_upload_validation_models_and_removed_endpoints(config):
    with TestClient(create_app(config, model=ScriptedModel(), embeddings=FakeEmbeddings())) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/agent/models").json()["models"]) == 1
        assert client.post("/auth/login").status_code == 404
        assert client.post("/agent/tts").status_code == 404
        assert (
            client.post("/agent/chat", data={"message": "你好", "model_provider": "local"}).status_code == 400
        )
        assert (
            client.post(
                "/agent/chat", data={"message": "看图"}, files={"image": ("a.png", b"bad")}
            ).status_code
            == 400
        )


async def test_busy_isolation_cancellation_and_timeout(config):
    class SlowModel(ScriptedModel):
        async def classify(self, messages, originals):
            await asyncio.sleep(5)
            return Intent()

    app = create_app(config, model=SlowModel(), embeddings=FakeEmbeddings())
    async with app.router.lifespan_context(app):
        service = app.state.service
        service.reserve("busy")

        async def sink(event, data):
            pass

        task = asyncio.create_task(service.run("busy", "hello", [], sink))
        await asyncio.sleep(0.05)
        with pytest.raises(Exception, match="正在处理"):
            service.reserve("busy")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "busy" not in service.busy
        assert (await service.store.conversation("busy"))["messages"][1]["status"] == "cancelled"
        assert await service.store.history_groups("busy") == []
        service.settings.turn_timeout_seconds = 0.02
        with pytest.raises(Exception, match="总时限"):
            await service.run(service.reserve("other"), "hello", [], sink)
        assert (await service.store.conversation("other"))["messages"][1]["status"] == "failed"


def test_history_trims_whole_tool_turns():
    from langchain_core.messages import HumanMessage, message_to_dict

    groups = [
        [message_to_dict(HumanMessage(content="x" * 1500))],
        [
            message_to_dict(HumanMessage(content="问")),
            message_to_dict(call("search_knowledge", {"query": "斗拱"})),
            message_to_dict(ToolMessage(content="资料", tool_call_id="call_search_knowledge")),
            message_to_dict(AIMessage(content="回答")),
        ],
    ]
    result = recent_history(groups, 1000)
    assert len(result) == 4
    assert isinstance(result[1], AIMessage) and isinstance(result[2], ToolMessage)


def test_cross_conversation_assets_and_old_batches_are_not_available(config, png):
    model = ScriptedModel(Intent())
    app = create_app(config, model=model, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        client.post(
            "/agent/chat",
            data={"message": "收下图片", "conversation_id": "first"},
            files={"image": ("old.png", png)},
        )
        previous = client.get("/agent/conversations/first").json()["messages"][0]["assets"][0]["id"]
        model.intent = Intent(kind="visual", operations=["match"])
        model.decisions = [lambda messages, tools: call("match_single_image", {"asset_id": previous})]
        result = client.post(
            "/agent/chat",
            data={"message": "匹配", "conversation_id": "second"},
            files={"image": ("new.png", png)},
        ).json()
        assert not result["tool_calls"][0]["success"]
        model.decisions = [lambda messages, tools: call("match_single_image", {"asset_id": previous})]
        result = client.post(
            "/agent/chat",
            data={"message": "匹配", "conversation_id": "first"},
            files={"image": ("newer.png", png)},
        ).json()
        assert not result["tool_calls"][0]["success"]
        assert "old.png" not in model.observations[-2][0].content


def test_upload_limits_and_no_history_shared_between_conversations(config, png):
    model = ScriptedModel(answer="hello")
    with TestClient(create_app(config, model=model, embeddings=FakeEmbeddings())) as client:
        assert (
            client.post("/agent/chat", data={"message": "secret", "conversation_id": "one"}).status_code
            == 200
        )
        assert (
            client.post("/agent/chat", data={"message": "hello", "conversation_id": "two"}).status_code == 200
        )
        assert "secret" not in str(model.classifications[-1])
        files = [("images", (f"{index}.png", png)) for index in range(6)]
        assert client.post("/agent/chat", data={"message": "看图"}, files=files).status_code == 400
        config.max_image_bytes = 1024
        response = client.post(
            "/agent/chat", data={"message": "看图"}, files={"image": ("big.png", png * 100)}
        )
        assert response.status_code == 413
        assert client.post("/agent/chat", data={"message": "   "}).status_code == 400


def test_temporary_failure_only_gets_one_retry(config):
    from app.errors import AgentError

    class FailingKnowledge:
        def __init__(self):
            self.calls = 0

        async def search(self, query):
            self.calls += 1
            raise AgentError("temporary", retryable=True)

    model = ScriptedModel(
        Intent(kind="knowledge"), [lambda messages, tools: call("search_knowledge", {"query": "q"})] * 3
    )
    app = create_app(config, model=model, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        failing = FailingKnowledge()
        app.state.service.agent_graph.toolbox.knowledge = failing
        result = client.post("/agent/chat", data={"message": "领域问题"}).json()
        assert failing.calls == 2
        assert len(result["tool_calls"]) == 2
        assert "无法给出有依据" in result["answer"]


def test_invalid_citation_and_provider_failure_are_not_saved_as_complete(config):
    model = ScriptedModel(Intent(kind="knowledge"), answer="某结论[99]")
    app = create_app(config, model=model, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        app.state.service.agent_graph.toolbox.knowledge = FakeKnowledge()
        response = client.post("/agent/stream", data={"message": "领域问题", "conversation_id": "invalid"})
        assert "event: error" in response.text
        assert "event: final\r\n" not in response.text
        assert client.get("/agent/conversations/invalid").json()["messages"][-1]["status"] == "failed"


def test_segment_then_repair_cannot_skip_dependency(config, png):
    model = ScriptedModel(
        Intent(kind="visual", operations=["segment", "repair"]),
        [lambda messages, tools: call("repair_image", {"asset_id": available(messages)[0]["id"]})],
    )
    backend = Backend(png)
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post(
            "/agent/chat", data={"message": "先分割后修复"}, files={"image": ("a.png", png)}
        ).json()
        assert not result["tool_calls"][0]["success"]
        assert not backend.calls


def test_non_dougong_image_can_decline_matching(config, png):
    model = ScriptedModel(
        Intent(kind="visual", operations=["match"]), answer="图中不是斗拱，不能给出斗拱匹配结果。"
    )
    backend = Backend(png)
    with TestClient(
        create_app(
            config, model=model, embeddings=FakeEmbeddings(), transport=httpx.MockTransport(backend.handle)
        )
    ) as client:
        result = client.post(
            "/agent/chat", data={"message": "识别这个的朝代"}, files={"image": ("a.png", png)}
        ).json()
        assert not backend.calls and not result["tool_calls"]
        assert "不是斗拱" in result["answer"]


async def test_sse_disconnect_cancels_unfinished_generation(config):
    from urllib.parse import urlencode

    class SlowAnswer(ScriptedModel):
        async def answer(self, messages):
            yield "正在回答"
            await asyncio.sleep(10)
            yield "不应继续"

    app = create_app(config, model=SlowAnswer(), embeddings=FakeEmbeddings())
    async with app.router.lifespan_context(app):
        body = urlencode({"message": "你好", "conversation_id": "disconnect"}).encode()
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": body, "more_body": False})
        sent = []

        async def receive():
            return await incoming.get()

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and b"event: answer_delta" in message.get("body", b""):
                await incoming.put({"type": "http.disconnect"})

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "http",
            "method": "POST",
            "path": "/agent/stream",
            "raw_path": b"/agent/stream",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/x-www-form-urlencoded"),
                (b"content-length", str(len(body)).encode()),
            ],
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=3)
        assert "disconnect" not in app.state.service.busy
        record = await app.state.service.store.conversation("disconnect")
        assert record["messages"][-1]["status"] == "cancelled"
        assert not any(b"event: final\r\n" in item.get("body", b"") for item in sent)


def test_application_without_credentials_starts_and_returns_configuration_error(config):
    config.deepseek_api_key = ""
    config.siliconflow_api_key = ""
    app = create_app(config)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and not health["deepseek_configured"]
        response = client.post("/agent/chat", data={"message": "你好", "conversation_id": "no_key"})
        assert response.status_code == 503
        assert "DEEPSEEK_API_KEY" in response.json()["detail"]
        assert not app.state.service.busy


async def test_old_stream_cleanup_does_not_unlock_a_new_request(config):
    from urllib.parse import urlencode

    app = create_app(config, model=ScriptedModel(), embeddings=FakeEmbeddings())
    async with app.router.lifespan_context(app):
        body = urlencode({"message": "你好", "conversation_id": "race"}).encode()
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": body, "more_body": False})
        reserved = False

        async def receive():
            return await incoming.get()

        async def send(message):
            nonlocal reserved
            if message["type"] == "http.response.body" and b"event: final\r\n" in message.get("body", b""):
                app.state.service.reserve("race")
                reserved = True

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "http",
            "method": "POST",
            "path": "/agent/stream",
            "raw_path": b"/agent/stream",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/x-www-form-urlencoded"),
                (b"content-length", str(len(body)).encode()),
            ],
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=3)
        assert reserved and "race" in app.state.service.busy
        app.state.service.busy.pop("race")
