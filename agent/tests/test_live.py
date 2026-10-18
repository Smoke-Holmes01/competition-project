"""Opt-in live checks; no credentials or provider exception bodies are printed."""

import os

import httpx
import numpy as np
import pytest
from langchain_core.messages import HumanMessage

from app.config import Settings
from app.providers import DeepSeekModel, Embeddings, with_images

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1", reason="真实联调须显式设置 RUN_LIVE_TESTS=1"
)


async def test_live_bge_m3():
    settings = Settings()
    if not settings.siliconflow_api_key:
        pytest.skip("SILICONFLOW_API_KEY 未配置")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            vectors = await Embeddings(settings, client).embed(["斗拱承托屋檐。"])
        assert vectors.shape == (1, 1024) and np.isclose(np.linalg.norm(vectors[0]), 1)
    except Exception as exc:  # noqa: BLE001 -- redact external error bodies and credentials
        pytest.fail(f"BGE-M3 live check failed: {type(exc).__name__}", pytrace=False)


async def test_live_deepseek_text_vision_tools(png):
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY 未配置")
    model = DeepSeekModel(settings)
    try:
        intent = await model.classify([HumanMessage(content="你好")], 0)
        assert intent.kind == "chat"
        message = with_images(
            HumanMessage(content="请简短描述图中颜色，不推测年代。"), [("test", png, "image/png")]
        )
        assert "".join([part async for part in model.answer([message])])
        schema = {
            "type": "function",
            "function": {
                "name": "preview_model",
                "description": "预览已有模型",
                "parameters": {
                    "type": "object",
                    "properties": {"model_id": {"type": "integer"}},
                    "required": ["model_id"],
                },
            },
        }
        decision = await model.decide(
            [HumanMessage(content="请调用 preview_model，参数 model_id 为 1，不要直接回答。")], [schema]
        )
        assert decision.tool_calls and decision.tool_calls[0]["args"]["model_id"] == 1
    except Exception as exc:  # noqa: BLE001 -- redact external error bodies and credentials
        pytest.fail(f"DeepSeek live check failed: {type(exc).__name__}", pytrace=False)
    finally:
        await model.aclose()


async def test_live_visual_service_health_and_model_resource():
    settings = Settings()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            health = await client.get(settings.backend_base_url.rstrip("/") + "/health")
            assert health.is_success
            model = await client.head(settings.backend_base_url.rstrip("/") + "/models_glb/1.glb")
            if model.status_code == 404:
                pytest.skip("视觉服务可达，但 1 号 GLB 缺失；真实图片推理仍须另行验证")
            assert model.is_success
    except httpx.RequestError:
        pytest.skip("视觉服务未启动或不可连接；未进行真实视觉推理")
