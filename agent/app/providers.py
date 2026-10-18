"""Small provider boundaries: tests replace these without changing graph code."""

import base64
import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx
import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.errors import AgentError
from app.schemas import Intent

INTENT_PROMPT = """你是古建筑工作台的请求分析器。仅分析用户当前的请求，不执行工具。
区分普通闲聊chat、领域事实knowledge、一般图片问答image、专用图片任务visual、编号预览preview。
operations只列用户明确要求的match/segment/repair/preview，并保留用户要求的先后顺序；普通识别不能授权segment或repair。
知识事实需要needs_knowledge=true；问候、图片可见内容描述不需要。
image_positions按最近上传批次从1编号；只有明确多图/综合/全部处理才all_images=true。
单图任务有多张可选且没有明确指代时，填clarification询问具体图片。不要猜编号。
先分割后修复需要segment和repair；引用上一工具的图时根据上下文判断，不扩大操作范围。
无图片的纯知识问题不能误判为匹配任务。"""


class Model(Protocol):
    async def classify(self, messages: list[BaseMessage], originals: int) -> Intent: ...
    async def decide(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage: ...
    def answer(self, messages: list[BaseMessage]) -> AsyncIterator[str]: ...


class DeepSeekModel:
    def __init__(self, settings: Settings, **client_options):
        self.settings = settings
        # Explicit vendor parameter: no hidden reasoning state to lose across checkpoints.
        self.client = ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key or "not-configured",
            api_base=settings.deepseek_base_url,
            extra_body={"thinking": {"type": "disabled"}},
            temperature=0,
            max_tokens=settings.output_tokens,
            timeout=settings.request_timeout_seconds,
            max_retries=0,
            **client_options,
        )

    def check(self):
        if not self.settings.deepseek_api_key:
            raise AgentError("请配置 DEEPSEEK_API_KEY 后重试。", status=503)

    async def classify(self, messages: list[BaseMessage], originals: int) -> Intent:
        self.check()
        classifier = self.client.with_structured_output(Intent, method="function_calling")
        return await classifier.ainvoke(
            [
                SystemMessage(content=INTENT_PROMPT + f"\n当前批次原图数量：{originals}。"),
                *messages,
            ]
        )

    async def decide(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        self.check()
        if not tools:
            return AIMessage(content="可以结束工具阶段。")
        return await self.client.bind_tools(tools).ainvoke(messages)

    async def answer(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        self.check()
        async for chunk in self.client.astream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    async def aclose(self):
        await self.client.root_async_client.close()
        self.client.root_client.close()


def with_images(message: HumanMessage, images: list[tuple[str, bytes, str]]) -> HumanMessage:
    """Create an ephemeral request copy; never checkpoint base64 or mutate saved messages."""
    content = [{"type": "text", "text": str(message.content)}]
    for asset_id, data, mime in images:
        content.append({"type": "text", "text": f"附件 asset_id={asset_id}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"},
            }
        )
    return HumanMessage(content=content, id=message.id)


class Embeddings:
    dimension = 1024

    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not self.settings.siliconflow_api_key:
            raise AgentError("请配置 SILICONFLOW_API_KEY 后构建或查询知识库。", status=503)
        if not texts or any(not text.strip() or len(text) > 2000 for text in texts):
            raise AgentError("嵌入输入必须非空且已切分为不超过 2000 字符的片段。")
        try:
            response = await self.client.post(
                self.settings.siliconflow_base_url.rstrip("/") + "/embeddings",
                headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"},
                json={"model": self.settings.embedding_model, "input": texts, "encoding_format": "float"},
            )
            response.raise_for_status()
            records = sorted(response.json()["data"], key=lambda row: row["index"])
            if [row["index"] for row in records] != list(range(len(texts))):
                raise ValueError("embedding indices do not match inputs")
            vectors = np.asarray([row["embedding"] for row in records], dtype=np.float32)
            if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
                raise ValueError("invalid embedding shape or values")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError("zero embedding")
            return vectors / norms
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise AgentError(
                f"嵌入 API 返回 HTTP {status}。", retryable=status == 429 or status >= 500, status=502
            ) from exc
        except httpx.RequestError as exc:
            raise AgentError("嵌入 API 暂时无法连接。", retryable=True, status=502) from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise AgentError("嵌入 API 返回的数据不完整或维度不符合 1024。", status=502) from exc


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
