import asyncio

from app.agent import ReactAgent
from app.config import Settings
from app.llm import ChatResult
from app.schemas import HistoryMessage, ToolResult


def test_agent_marks_deep_image_question_as_missing_image():
    agent = ReactAgent(Settings())
    result = asyncio.run(agent.run(message="这个斗拱是哪个朝代的？"))

    assert result.tool_called is False
    assert any(item.event == "missing_image" for item in result.status_trace)
    assert "上传" in result.answer


def test_agent_simple_image_question_skips_tool_and_specific_rag():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))

    async def fake_vision_chat(**kwargs):
        return ChatResult(content="图片里是一栋房子。", tool_calls=[], raw={})

    agent.llm.vision_chat = fake_vision_chat
    result = asyncio.run(agent.run(message="这张图是什么？", image_bytes=b"fake-image"))

    assert result.tool_called is False
    assert result.render_instruction is None
    assert result.rag_hits == []
    assert result.answer == "图片里是一栋房子。"
    assert "retrieving" not in [item.event for item in result.status_trace]


def test_agent_simple_image_question_uses_general_vision_not_match_tool():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))
    captured = {}

    async def fake_vision_chat(**kwargs):
        captured.update(kwargs)
        return ChatResult(content="图片里是一栋房子。", tool_calls=[], raw={})

    async def fail_match(*args, **kwargs):
        raise AssertionError("general image question should not call dougong match tool")

    agent.llm.vision_chat = fake_vision_chat
    agent.tools.match_uploaded_image = fail_match
    result = asyncio.run(agent.run(message="你能告诉我这张图片有什么内容吗", image_bytes=b"fake-image"))

    assert result.tool_called is False
    assert result.render_instruction is None
    assert result.answer == "图片里是一栋房子。"
    assert captured["image_bytes"] == b"fake-image"


def test_agent_deep_non_dougong_image_skips_match_tool():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))

    async def fake_vision_chat(**kwargs):
        return ChatResult(content="NO", tool_calls=[], raw={})

    async def fail_match(*args, **kwargs):
        raise AssertionError("non-dougong image should not call dougong match tool")

    agent.llm.vision_chat = fake_vision_chat
    agent.tools.match_uploaded_image = fail_match
    result = asyncio.run(agent.run(message="这个是哪个朝代的？", image_bytes=b"fake-image"))

    assert result.tool_called is False
    assert result.render_instruction is None
    assert "目前系统只支持对斗拱图片识别朝代" in result.answer


def test_agent_deep_image_question_returns_glb_render_instruction():
    agent = ReactAgent(Settings())

    async def fake_predict_single_image(*args, **kwargs):
        return ToolResult(
            success=True,
            obj_id=1,
            final_group="group_001",
            obj_path="models/1.obj",
            obj_url="/models/1.obj",
            glb_path="models_glb/1.glb",
            glb_url="/models_glb/1.glb",
            model_name="清代官式建筑中高等级的七踩斗拱",
            dynasty="清代",
            description="测试说明",
            knowledge_id="knowledge_001",
            confidence=0.9,
        )

    agent.tools.predict_single_image = fake_predict_single_image
    result = asyncio.run(agent.run(message="这个斗拱是哪个朝代的？", image_bytes=b"fake-image"))

    assert result.tool_called is True
    assert result.render_instruction is not None
    assert result.render_instruction.model_format == "glb"
    assert result.render_instruction.model_url == "/models_glb/1.glb"
    assert "127.0.0.1" not in result.render_instruction.model_url


def test_agent_preview_model_by_group_number_without_image():
    agent = ReactAgent(Settings())

    result = asyncio.run(agent.run(message="我想看看第33组模型"))

    assert result.tool_called is True
    assert result.tool_calls[0].tool == "preview_model"
    assert result.render_instruction is not None
    assert result.render_instruction.model_id == 33
    assert result.render_instruction.model_url == "/models_glb/33.glb"
    assert result.ui_instruction is not None
    assert result.ui_instruction.target_view == "model"


def test_agent_segment_image_only_calls_segment_tool():
    agent = ReactAgent(Settings())
    calls = []

    async def fake_segment_image(*args, **kwargs):
        calls.append("segment")
        return {
            "ok": True,
            "payload": {
                "success": True,
                "crop_url": "/outputs/crop.png",
                "mask_url": "/outputs/mask.png",
                "message": "分割成功",
            },
        }

    async def fail_predict(*args, **kwargs):
        raise AssertionError("segment request should not call predict")

    agent.tools.segment_image = fake_segment_image
    agent.tools.predict_single_image = fail_predict

    result = asyncio.run(agent.run(message="请给我将这张斗拱图片分割出来", image_bytes=b"fake-image"))

    assert calls == ["segment"]
    assert result.tool_calls[0].tool == "segment_image"
    assert [artifact.label for artifact in result.artifacts] == ["分割主体裁剪图", "分割 Mask"]
    assert result.ui_instruction is not None
    assert result.ui_instruction.target_view == "recognition"


def test_agent_reuses_cached_image_for_later_tool_request():
    agent = ReactAgent(Settings())
    captured = {}

    async def fake_segment_image(image_bytes, **kwargs):
        captured["image_bytes"] = image_bytes
        return {
            "ok": True,
            "payload": {
                "success": True,
                "crop_url": "/outputs/crop.png",
                "mask_url": "/outputs/mask.png",
            },
        }

    agent.tools.segment_image = fake_segment_image

    asyncio.run(agent.run(message="先记住这张图", conversation_id="conv_cache", image_bytes=b"cached-image"))
    result = asyncio.run(agent.run(message="把刚才那张图分割出来", conversation_id="conv_cache"))

    assert captured["image_bytes"] == b"cached-image"
    assert result.tool_calls[0].tool == "segment_image"


def test_agent_multi_image_match_calls_multi_tool():
    agent = ReactAgent(Settings())
    captured = {}

    async def fake_predict_multi_images(images):
        captured["count"] = len(images)
        return ToolResult(
            success=True,
            obj_id=2,
            final_group="group_002",
            glb_url="/models_glb/2.glb",
            confidence=0.8,
        )

    agent.tools.predict_multi_images = fake_predict_multi_images
    result = asyncio.run(
        agent.run(
            message="请对这几张图做融合匹配",
            images=[
                {"bytes": b"one", "filename": "1.png", "content_type": "image/png"},
                {"bytes": b"two", "filename": "2.png", "content_type": "image/png"},
            ],
        )
    )

    assert captured["count"] == 2
    assert result.tool_calls[0].tool == "match_multi_image"
    assert result.render_instruction is not None
    assert result.render_instruction.model_id == 2


def test_agent_invalid_model_id_does_not_call_tool():
    agent = ReactAgent(Settings())

    result = asyncio.run(agent.run(message="我想看看第99组模型"))

    assert result.tool_called is False
    assert result.tool_calls[0].success is False
    assert "1 到 46" in result.answer


def test_agent_passes_history_to_llm_prompt():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))
    agent.rag.search = lambda query: []
    captured_messages = []

    async def fake_chat(messages, **kwargs):
        captured_messages.append(list(messages))
        return ChatResult(content="这是基于前文的回答。", tool_calls=[], raw={})

    agent.llm.chat = fake_chat
    result = asyncio.run(
        agent.run(
            message="它有什么特点？",
            conversation_id="conv_existing",
            history=[
                HistoryMessage(role="user", content="这个斗拱是哪个朝代的？"),
                HistoryMessage(role="assistant", content="这是清代官式七踩斗拱。"),
            ],
        )
    )

    assert result.conversation_id == "conv_existing"
    prompt_messages = captured_messages[-1]
    assert prompt_messages[1] == {"role": "user", "content": "这个斗拱是哪个朝代的？"}
    assert prompt_messages[2] == {"role": "assistant", "content": "这是清代官式七踩斗拱。"}
    assert result.answer == "这是基于前文的回答。"


def test_agent_new_conversation_does_not_inherit_old_history():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))
    agent.rag.search = lambda query: []
    captured_messages = []

    async def fake_chat(messages, **kwargs):
        captured_messages.append(list(messages))
        return ChatResult(content="ok", tool_calls=[], raw={})

    agent.llm.chat = fake_chat
    asyncio.run(
        agent.run(
            message="继续说",
            conversation_id="conv_old",
            history=[HistoryMessage(role="assistant", content="旧对话内容")],
        )
    )
    asyncio.run(agent.run(message="新问题", conversation_id="conv_new", history=[]))

    latest_payload = "\n".join(item["content"] for item in captured_messages[-1])
    assert "旧对话内容" not in latest_payload


def test_agent_crops_history_before_llm_prompt():
    agent = ReactAgent(Settings(siliconflow_base_url="http://llm.example/v1", siliconflow_model="model"))
    agent.rag.search = lambda query: []
    captured_messages = []

    async def fake_chat(messages, **kwargs):
        captured_messages.append(list(messages))
        return ChatResult(content="ok", tool_calls=[], raw={})

    agent.llm.chat = fake_chat
    history = [
        HistoryMessage(role="user" if index % 2 == 0 else "assistant", content=f"{index:02d}-" + ("x" * 500))
        for index in range(25)
    ]

    asyncio.run(agent.run(message="总结一下", conversation_id="conv_long", history=history))

    history_messages = captured_messages[-1][1:-1]
    assert len(history_messages) <= 20
    assert sum(len(item["content"]) for item in history_messages) <= 8000
    assert "00-" not in "\n".join(item["content"] for item in history_messages)


def test_agent_uses_local_llm_when_requested():
    agent = ReactAgent(Settings())
    agent.rag.search = lambda query: []

    class FakeLocalLlm:
        configured = True

        async def chat(self, messages, **kwargs):
            return ChatResult(content="本地模型回答。", tool_calls=[], raw={})

        async def stream_chat(self, messages, **kwargs):
            yield "本地模型回答。"

    agent.local_llm = FakeLocalLlm()
    result = asyncio.run(agent.run(message="斗拱是什么？", model_provider="local"))

    assert result.answer == "本地模型回答。"
