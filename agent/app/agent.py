from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from app.config import Settings
from app.intent import is_image_specific_question, is_simple_image_question, should_call_match_tool
from app.llm import ChatClient, LLMError, LocalTransformersChatClient, SiliconFlowChatClient
from app.rag import PdfKnowledgeBase
from app.schemas import (
    AgentArtifact,
    AgentChatResponse,
    AgentToolCall,
    HistoryMessage,
    RagHit,
    RenderInstruction,
    StatusEvent,
    ToolResult,
    UiInstruction,
)
from app.tools import VisionToolClient, build_render_instruction


StatusSink = Callable[[StatusEvent], Awaitable[None]]
AnswerSink = Callable[[str], Awaitable[None]]

MAX_MODEL_ID = 46
IMAGE_CACHE_TTL_SECONDS = 30 * 60
IMAGE_CACHE_LIMIT = 5
IMAGE_TOOL_NAMES = {
    "segment_image",
    "repair_image",
    "repair_segmented_image",
    "match_single_image",
    "match_multi_image",
}


@dataclass
class CachedImage:
    bytes: bytes
    filename: str
    content_type: str
    created_at: float


@dataclass
class ToolPlan:
    name: str
    model_id: Optional[int] = None


class ReactAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = SiliconFlowChatClient(settings)
        self.local_llm = LocalTransformersChatClient(settings)
        self.rag = PdfKnowledgeBase(settings)
        self.tools = VisionToolClient(settings)
        self._image_cache: Dict[str, List[CachedImage]] = {}

    async def run(
        self,
        *,
        message: str,
        conversation_id: str = "",
        history: Optional[List[HistoryMessage]] = None,
        image_bytes: Optional[bytes] = None,
        image_filename: str = "upload.png",
        image_content_type: str = "application/octet-stream",
        images: Optional[List[Dict[str, Any]]] = None,
        model_provider: str = "api",
        status_sink: Optional[StatusSink] = None,
        answer_sink: Optional[AnswerSink] = None,
    ) -> AgentChatResponse:
        llm = self._get_llm_client(model_provider)
        normalized_conversation_id = conversation_id.strip() or f"conv_{uuid4().hex}"
        prepared_history = self._prepare_history(history or [])
        status_trace: List[StatusEvent] = []

        async def record(event: str, text: str) -> None:
            status = StatusEvent(event=event, message=text)
            status_trace.append(status)
            if status_sink:
                await status_sink(status)

        await record("analyzing", "正在分析问题意图")

        current_images = self._normalize_request_images(
            images=images,
            image_bytes=image_bytes,
            image_filename=image_filename,
            image_content_type=image_content_type,
        )
        if current_images:
            self._store_images(normalized_conversation_id, current_images)

        cached_images = self._get_cached_images(normalized_conversation_id)
        available_images = current_images or cached_images
        has_image = bool(available_images)

        plan = await self._plan_tool(message=message, has_image=has_image)
        if plan and plan.name in IMAGE_TOOL_NAMES and not available_images:
            await record("missing_image", "该工具需要图片，但当前会话没有可复用的上传图片")
            answer = "这个操作需要图片。请先上传斗拱图片后再试；如果服务重启过，之前的临时图片缓存也需要重新上传。"
            if answer_sink:
                await answer_sink(answer)
            await record("final", "回答生成完成")
            return AgentChatResponse(
                conversation_id=normalized_conversation_id,
                answer=answer,
                tool_called=False,
                tool_result=None,
                rag_hits=[],
                render_instruction=None,
                tool_calls=[],
                artifacts=[],
                ui_instruction=None,
                status_trace=status_trace,
            )

        if plan and plan.model_id is not None and not self._is_valid_model_id(plan.model_id):
            answer = f"模型编号 {plan.model_id} 不在当前资源范围内；请使用 1 到 {MAX_MODEL_ID} 之间的组号或模型 ID。"
            if answer_sink:
                await answer_sink(answer)
            await record("final", "回答生成完成")
            return AgentChatResponse(
                conversation_id=normalized_conversation_id,
                answer=answer,
                tool_called=False,
                tool_result=None,
                rag_hits=[],
                render_instruction=None,
                tool_calls=[
                    AgentToolCall(
                        tool=plan.name,
                        success=False,
                        error=answer,
                    )
                ],
                artifacts=[],
                ui_instruction=None,
                status_trace=status_trace,
            )

        if has_image and not plan and is_image_specific_question(message):
            await record("finalizing", "正在识别图片内容")
            first = available_images[0]
            answer = await self._answer_general_image_question(
                message=message,
                image_bytes=first.bytes,
                image_filename=first.filename,
                image_content_type=first.content_type,
            )
            if answer_sink:
                await answer_sink(answer)
            await record("final", "回答生成完成")
            return AgentChatResponse(
                conversation_id=normalized_conversation_id,
                answer=answer,
                tool_called=False,
                tool_result=None,
                rag_hits=[],
                render_instruction=None,
                tool_calls=[],
                artifacts=[],
                ui_instruction=UiInstruction(target_view="recognition", message="已完成通用图片识别"),
                status_trace=status_trace,
            )

        if plan:
            return await self._run_planned_tool(
                plan=plan,
                message=message,
                conversation_id=normalized_conversation_id,
                history=prepared_history,
                images=available_images,
                llm=llm,
                record=record,
                answer_sink=answer_sink,
                status_trace=status_trace,
            )

        rag_hits = await self._search_rag(message, record)
        await record("finalizing", "正在组织最终回答")
        answer = await self._compose_answer(
            message=message,
            history=prepared_history,
            tool_result=None,
            rag_hits=rag_hits,
            llm=llm,
            answer_sink=answer_sink,
        )
        await record("final", "回答生成完成")

        return AgentChatResponse(
            conversation_id=normalized_conversation_id,
            answer=answer,
            tool_called=False,
            tool_result=None,
            rag_hits=rag_hits,
            render_instruction=None,
            tool_calls=[],
            artifacts=[],
            ui_instruction=UiInstruction(target_view="knowledge") if rag_hits else None,
            status_trace=status_trace,
        )

    async def _run_planned_tool(
        self,
        *,
        plan: ToolPlan,
        message: str,
        conversation_id: str,
        history: List[HistoryMessage],
        images: List[CachedImage],
        llm: ChatClient,
        record: Callable[[str, str], Awaitable[None]],
        answer_sink: Optional[AnswerSink],
        status_trace: List[StatusEvent],
    ) -> AgentChatResponse:
        await record("calling_tool", f"正在调用工具：{plan.name}")
        tool_calls: List[AgentToolCall] = []
        artifacts: List[AgentArtifact] = []
        tool_result: Optional[ToolResult] = None
        render_instruction: Optional[RenderInstruction] = None
        ui_instruction: Optional[UiInstruction] = None
        rag_hits: List[RagHit] = []

        if plan.name == "preview_model":
            assert plan.model_id is not None
            tool_result = self.tools.preview_model(plan.model_id)
            render_instruction = build_render_instruction(tool_result)
            artifacts.append(self._model_artifact(tool_result, "preview_model"))
            tool_calls.append(self._tool_call_from_result("preview_model", tool_result))
            ui_instruction = UiInstruction(target_view="model", active_artifact_id=artifacts[0].id)
            answer = f"已切换到第 {plan.model_id} 组模型，工作台会使用 {tool_result.glb_url} 进行 GLB 预览。"

        elif plan.name == "segment_image":
            first = images[0]
            data = await self.tools.segment_image(first.bytes, filename=first.filename, content_type=first.content_type)
            artifacts = self._artifacts_from_segment(data)
            tool_calls.append(self._tool_call_from_payload("segment_image", data))
            ui_instruction = UiInstruction(
                target_view="recognition",
                active_artifact_id=artifacts[0].id if artifacts else None,
            )
            answer = self._answer_from_payload(
                data,
                success_text="已完成图像分割，结果里包含主体裁剪图和 mask，可在工作台预览或下载。",
                failure_text="图像分割失败",
            )

        elif plan.name in {"repair_image", "repair_segmented_image"}:
            first = images[0]
            segmented = plan.name == "repair_segmented_image"
            data = await self.tools.repair_image(
                first.bytes,
                filename=first.filename,
                content_type=first.content_type,
                segmented=segmented,
            )
            artifacts = self._artifacts_from_repair(data, source_tool=plan.name)
            tool_calls.append(self._tool_call_from_payload(plan.name, data))
            ui_instruction = UiInstruction(
                target_view="recognition",
                active_artifact_id=artifacts[0].id if artifacts else None,
            )
            answer = self._answer_from_payload(
                data,
                success_text="已完成图像修复，修复结果可在工作台预览或下载。",
                failure_text="图像修复失败",
            )

        elif plan.name == "match_multi_image":
            payload_images = [
                {"bytes": item.bytes, "filename": item.filename, "content_type": item.content_type}
                for item in images
            ]
            tool_result = await self.tools.predict_multi_images(payload_images)
            render_instruction = build_render_instruction(tool_result)
            if tool_result.success:
                artifacts.append(self._model_artifact(tool_result, "match_multi_image"))
            tool_calls.append(self._tool_call_from_result("match_multi_image", tool_result))
            ui_instruction = UiInstruction(
                target_view="recognition",
                active_artifact_id=artifacts[0].id if artifacts else None,
            )
            rag_hits = await self._search_rag(self._build_rag_query(message, tool_result), record)
            answer = await self._compose_answer(
                message=message,
                history=history,
                tool_result=tool_result,
                rag_hits=rag_hits,
                llm=llm,
                answer_sink=None,
            )

        elif plan.name == "match_single_image":
            first = images[0]
            is_dougong = await self._image_likely_dougong(
                message=message,
                image_bytes=first.bytes,
                image_filename=first.filename,
                image_content_type=first.content_type,
            )
            if is_dougong is False:
                answer = "这张图片看起来不像斗拱构件。目前系统只支持对斗拱图片识别朝代、匹配模型编号和生成 3D 资源。"
                tool_calls.append(
                    AgentToolCall(
                        tool="match_single_image",
                        success=False,
                        message="已跳过斗拱专用匹配工具",
                        error="图片主体不像斗拱",
                    )
                )
                if answer_sink:
                    await answer_sink(answer)
                await record("unsupported_image", "当前图片不像斗拱，已跳过斗拱专用匹配工具")
                await record("final", "回答生成完成")
                return AgentChatResponse(
                    conversation_id=conversation_id,
                    answer=answer,
                    tool_called=False,
                    tool_result=None,
                    rag_hits=[],
                    render_instruction=None,
                    tool_calls=tool_calls,
                    artifacts=[],
                    ui_instruction=UiInstruction(target_view="recognition"),
                    status_trace=status_trace,
                )
            tool_result = await self.tools.predict_single_image(
                first.bytes,
                filename=first.filename,
                content_type=first.content_type,
            )
            render_instruction = build_render_instruction(tool_result)
            if tool_result.success:
                artifacts.append(self._model_artifact(tool_result, "match_single_image"))
            tool_calls.append(self._tool_call_from_result("match_single_image", tool_result))
            ui_instruction = UiInstruction(
                target_view="recognition",
                active_artifact_id=artifacts[0].id if artifacts else None,
            )
            rag_hits = await self._search_rag(self._build_rag_query(message, tool_result), record)
            answer = await self._compose_answer(
                message=message,
                history=history,
                tool_result=tool_result,
                rag_hits=rag_hits,
                llm=llm,
                answer_sink=None,
            )

        else:
            rag_hits = await self._search_rag(message, record)
            answer = _summarize_rag_hits(rag_hits) or "我没有识别到需要调用的工具，已改为检索知识库。"
            ui_instruction = UiInstruction(target_view="knowledge") if rag_hits else None

        if answer_sink:
            await answer_sink(answer)
        await record("tool_done", f"工具处理完成：{plan.name}")
        await record("final", "回答生成完成")

        return AgentChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            tool_called=bool(tool_calls),
            tool_result=tool_result,
            rag_hits=rag_hits,
            render_instruction=render_instruction,
            tool_calls=tool_calls,
            artifacts=artifacts,
            ui_instruction=ui_instruction,
            status_trace=status_trace,
        )

    def _get_llm_client(self, model_provider: str) -> ChatClient:
        if model_provider == "local":
            return self.local_llm
        return self.llm

    async def _plan_tool(self, *, message: str, has_image: bool) -> Optional[ToolPlan]:
        rule_plan = self._plan_tool_by_rules(message=message, has_image=has_image)
        if rule_plan:
            return rule_plan
        if is_simple_image_question(message):
            return None
        return await self._plan_tool_with_llm(message)

    def _plan_tool_by_rules(self, *, message: str, has_image: bool) -> Optional[ToolPlan]:
        text = message.strip()
        lower = text.lower()
        model_id = self._extract_model_id(text)

        if model_id is not None and any(word in text for word in ("模型", "预览", "看看", "打开", "展示", "渲染", "组")):
            return ToolPlan("preview_model", model_id=model_id)

        if any(word in text for word in ("分割后修复", "先分割", "分割并修复", "抠出来再修复")):
            return ToolPlan("repair_segmented_image")

        if any(word in text for word in ("分割", "切割", "抠图", "抠出来", "主体裁剪", "mask", "掩膜")):
            return ToolPlan("segment_image")

        if any(word in text for word in ("修复", "补全", "复原", "去噪")):
            return ToolPlan("repair_image")

        match_requested = should_call_match_tool(text, has_image=True) or any(
            word in text for word in ("识别", "判断", "哪一组", "哪组", "分类", "融合匹配")
        )
        multi_requested = any(word in text for word in ("多图", "多张", "融合", "综合"))
        if match_requested and multi_requested:
            return ToolPlan("match_multi_image")
        if match_requested:
            return ToolPlan("match_single_image")

        if is_simple_image_question(text):
            return None
        return None

    async def _plan_tool_with_llm(self, message: str) -> Optional[ToolPlan]:
        if not self.llm.configured:
            return None

        system_prompt = (
            "你是斗拱工作台的工具规划器。只返回 JSON，不要解释。"
            "tool 只能是 none, preview_model, segment_image, repair_image, repair_segmented_image, "
            "match_single_image, match_multi_image。"
            "model_id 只能在用户明确给出 1 到 46 的模型编号、组号或 group 编号时填写，否则为 null。"
        )
        try:
            result = await self.llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.0,
                max_tokens=120,
            )
        except LLMError:
            return None

        try:
            payload = json.loads(result.content.strip())
        except json.JSONDecodeError:
            return None

        tool = str(payload.get("tool") or "none")
        if tool == "none" or tool not in {
            "preview_model",
            "segment_image",
            "repair_image",
            "repair_segmented_image",
            "match_single_image",
            "match_multi_image",
        }:
            return None
        model_id = payload.get("model_id")
        return ToolPlan(tool, model_id=int(model_id) if isinstance(model_id, int) else None)

    async def _search_rag(self, query: str, record: Callable[[str, str], Awaitable[None]]) -> List[RagHit]:
        await record("retrieving", "正在检索知识库")
        return await asyncio.to_thread(self.rag.search, query)

    async def _image_likely_dougong(
        self,
        *,
        message: str,
        image_bytes: bytes,
        image_filename: str,
        image_content_type: str,
    ) -> Optional[bool]:
        if not self.llm.configured:
            return None

        try:
            result = await self.llm.vision_chat(
                text=(
                    "请判断图片主体是否是中国古建筑斗拱/斗栱/bracket set。"
                    "只回答 YES 或 NO，不要解释。用户问题：" + message
                ),
                image_bytes=image_bytes,
                filename=image_filename,
                content_type=image_content_type,
                system_prompt="你是图像分类器。只输出 YES 或 NO。",
                temperature=0.0,
                max_tokens=16,
            )
        except LLMError:
            return None

        answer = result.content.strip().lower()
        if answer.startswith("yes") or answer.startswith("是"):
            return True
        if answer.startswith("no") or answer.startswith("否") or "不是" in answer:
            return False
        return None

    async def _compose_answer(
        self,
        *,
        message: str,
        history: List[HistoryMessage],
        tool_result: Optional[ToolResult],
        rag_hits: List[RagHit],
        llm: ChatClient,
        answer_sink: Optional[AnswerSink] = None,
    ) -> str:
        if llm.configured:
            context = "\n\n".join(
                f"[知识库 {hit.source} 第{hit.page}页 score={hit.score:.3f}]\n{hit.text}"
                for hit in rag_hits
            )
            tool_payload = tool_result.model_dump(mode="json") if tool_result else None
            system_prompt = (
                "你是古建筑斗拱智能体。请基于用户问题、工具返回和知识库片段回答。"
                "不要输出隐藏推理链。不要编造知识库没有提供的朝代、名称或历史细节。"
            )
            user_prompt = {
                "question": message,
                "tool_result": tool_payload,
                "knowledge_context": context,
            }
            messages = [
                {"role": "system", "content": system_prompt},
                *[{"role": item.role, "content": item.content} for item in history],
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ]
            if answer_sink:
                try:
                    streamed_parts: List[str] = []
                    async for token in llm.stream_chat(messages, temperature=0.2, max_tokens=900):
                        streamed_parts.append(token)
                        await answer_sink(token)
                    streamed_answer = "".join(streamed_parts).strip()
                    if streamed_answer:
                        return streamed_answer
                except LLMError as exc:
                    if llm is self.local_llm:
                        answer = f"本地模型暂不可用：{exc}"
                        await answer_sink(answer)
                        return answer

            try:
                result = await llm.chat(messages, temperature=0.2, max_tokens=900)
                if result.content.strip():
                    return result.content.strip()
            except LLMError as exc:
                if llm is self.local_llm:
                    return f"本地模型暂不可用：{exc}"

        return self._fallback_answer(tool_result=tool_result, rag_hits=rag_hits)

    async def _answer_general_image_question(
        self,
        *,
        message: str,
        image_bytes: Optional[bytes],
        image_filename: str,
        image_content_type: str,
    ) -> str:
        if not image_bytes:
            return "请先上传图片，我才能识别图像内容。"
        if not self.llm.configured:
            return "我目前无法调用通用视觉模型来判断图片内容。请配置 API 模型后再试。"

        try:
            result = await self.llm.vision_chat(
                text=(
                    "请直接识别这张图片的主要内容，用中文简洁回答。"
                    "如果它不是斗拱，不要把它说成斗拱。用户问题：" + message
                ),
                image_bytes=image_bytes,
                filename=image_filename,
                content_type=image_content_type,
                system_prompt="你是通用图像理解助手。只回答可见内容，不要猜测不可见细节。",
                temperature=0.0,
                max_tokens=300,
            )
        except LLMError as exc:
            return f"我暂时无法调用通用视觉模型识别这张图片：{exc}"

        return result.content.strip() or "我暂时无法识别这张图片的具体内容。"

    def _fallback_answer(self, *, tool_result: Optional[ToolResult], rag_hits: List[RagHit]) -> str:
        if tool_result is not None:
            if not tool_result.success:
                return f"工具调用失败：{tool_result.error or '未知错误'}。"

            kb_summary = _summarize_rag_hits(rag_hits)
            kb_text = f"\n\n知识库中可参考的信息：{kb_summary}" if kb_summary else ""
            dynasty_text = f"，所属朝代为{tool_result.dynasty}" if tool_result.dynasty else ""
            name_text = f"，类型为{tool_result.model_name}" if tool_result.model_name else ""
            confidence_text = (
                f"，置信度 {tool_result.confidence:.4f}"
                if isinstance(tool_result.confidence, (int, float))
                else ""
            )
            return (
                f"匹配结果为 {tool_result.final_group or '未知分组'}，"
                f"对应模型 ID 为 {tool_result.obj_id}{name_text}{dynasty_text}{confidence_text}。"
                f"模型地址：{tool_result.glb_url or tool_result.obj_url or '未返回'}。"
                f"{kb_text}"
            )

        kb_summary = _summarize_rag_hits(rag_hits)
        if kb_summary:
            return kb_summary
        return "知识库中没有检索到足够相关的信息。请补充更具体的问题，或上传斗拱图片让我进行处理。"

    def _normalize_request_images(
        self,
        *,
        images: Optional[List[Dict[str, Any]]],
        image_bytes: Optional[bytes],
        image_filename: str,
        image_content_type: str,
    ) -> List[CachedImage]:
        normalized: List[CachedImage] = []
        now = time.time()
        for image in images or []:
            value = image.get("bytes")
            if isinstance(value, bytes) and value:
                normalized.append(
                    CachedImage(
                        bytes=value,
                        filename=str(image.get("filename") or f"upload_{len(normalized) + 1}.png"),
                        content_type=str(image.get("content_type") or "application/octet-stream"),
                        created_at=now,
                    )
                )
        if not normalized and image_bytes:
            normalized.append(
                CachedImage(
                    bytes=image_bytes,
                    filename=image_filename,
                    content_type=image_content_type,
                    created_at=now,
                )
            )
        return normalized

    def _store_images(self, conversation_id: str, images: List[CachedImage]) -> None:
        self._purge_expired_images()
        self._image_cache[conversation_id] = images[-IMAGE_CACHE_LIMIT:]

    def _get_cached_images(self, conversation_id: str) -> List[CachedImage]:
        self._purge_expired_images()
        return list(self._image_cache.get(conversation_id, []))

    def _purge_expired_images(self) -> None:
        now = time.time()
        expired = []
        for conversation_id, images in self._image_cache.items():
            fresh = [item for item in images if now - item.created_at <= IMAGE_CACHE_TTL_SECONDS]
            if fresh:
                self._image_cache[conversation_id] = fresh[-IMAGE_CACHE_LIMIT:]
            else:
                expired.append(conversation_id)
        for conversation_id in expired:
            self._image_cache.pop(conversation_id, None)

    def _extract_model_id(self, message: str) -> Optional[int]:
        patterns = [
            r"group[_\-\s]*(\d{1,3})",
            r"第\s*(\d{1,3})\s*组",
            r"(\d{1,3})\s*组",
            r"第\s*(\d{1,3})\s*(?:个)?模型",
            r"(\d{1,3})\s*号模型",
            r"模型\s*(\d{1,3})",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _is_valid_model_id(self, model_id: int) -> bool:
        return 1 <= model_id <= MAX_MODEL_ID

    def _build_rag_query(self, message: str, tool_result: Optional[ToolResult]) -> str:
        parts = [message]
        if tool_result and tool_result.obj_id is not None:
            parts.append(f"obj_id {tool_result.obj_id}")
        if tool_result and tool_result.final_group:
            parts.append(tool_result.final_group)
        return " ".join(parts)

    def _prepare_history(self, history: List[HistoryMessage]) -> List[HistoryMessage]:
        cropped = history[-20:]
        selected: List[HistoryMessage] = []
        remaining_chars = 8000
        for item in reversed(cropped):
            content = item.content.strip()
            if not content:
                continue
            if len(content) > remaining_chars:
                content = content[-remaining_chars:]
            if content:
                selected.append(HistoryMessage(role=item.role, content=content))
                remaining_chars -= len(content)
            if remaining_chars <= 0:
                break
        return list(reversed(selected))

    def _tool_call_from_payload(self, tool: str, data: Dict[str, Any]) -> AgentToolCall:
        payload = data.get("payload") or {}
        success = bool(data.get("ok")) and bool(payload.get("success", True))
        return AgentToolCall(
            tool=tool,
            success=success,
            message=str(payload.get("message") or ("调用成功" if success else "调用失败")),
            result=payload,
            error=None if success else str(data.get("error") or payload.get("message") or "工具调用失败"),
        )

    def _tool_call_from_result(self, tool: str, result: ToolResult) -> AgentToolCall:
        return AgentToolCall(
            tool=tool,
            success=result.success,
            message="调用成功" if result.success else "调用失败",
            result=result.model_dump(mode="json"),
            error=result.error,
        )

    def _artifacts_from_segment(self, data: Dict[str, Any]) -> List[AgentArtifact]:
        payload = data.get("payload") or {}
        artifacts: List[AgentArtifact] = []
        if payload.get("crop_url"):
            artifacts.append(
                AgentArtifact(
                    id="segment-crop",
                    type="image",
                    label="分割主体裁剪图",
                    url=payload.get("crop_url"),
                    preview_url=payload.get("crop_url"),
                    download_url=payload.get("crop_url"),
                    mime_type="image/png",
                    source_tool="segment_image",
                )
            )
        if payload.get("mask_url"):
            artifacts.append(
                AgentArtifact(
                    id="segment-mask",
                    type="image",
                    label="分割 Mask",
                    url=payload.get("mask_url"),
                    preview_url=payload.get("mask_url"),
                    download_url=payload.get("mask_url"),
                    mime_type="image/png",
                    source_tool="segment_image",
                )
            )
        return artifacts

    def _artifacts_from_repair(self, data: Dict[str, Any], *, source_tool: str) -> List[AgentArtifact]:
        payload = data.get("payload") or {}
        repaired_url = payload.get("repaired_url")
        if not repaired_url:
            return []
        return [
            AgentArtifact(
                id=f"{source_tool}-repaired",
                type="image",
                label="修复结果图",
                url=repaired_url,
                preview_url=repaired_url,
                download_url=repaired_url,
                mime_type="image/png",
                source_tool=source_tool,
            )
        ]

    def _model_artifact(self, tool_result: ToolResult, source_tool: str) -> AgentArtifact:
        model_id = tool_result.obj_id or 0
        url = tool_result.glb_url or tool_result.obj_url
        return AgentArtifact(
            id=f"model-{model_id}",
            type="model",
            label=f"模型 {model_id} GLB 资源",
            url=url,
            preview_url=url,
            download_url=url,
            mime_type="model/gltf-binary" if tool_result.glb_url else "model/obj",
            source_tool=source_tool,
        )

    def _answer_from_payload(self, data: Dict[str, Any], *, success_text: str, failure_text: str) -> str:
        payload = data.get("payload") or {}
        success = bool(data.get("ok")) and bool(payload.get("success", True))
        if success:
            links = []
            for key in ("crop_url", "mask_url", "repaired_url"):
                if payload.get(key):
                    links.append(str(payload[key]))
            link_text = f" 结果链接：{'，'.join(links)}。" if links else ""
            return f"{success_text}{link_text}"
        return f"{failure_text}：{data.get('error') or payload.get('message') or '未知错误'}"


def _summarize_rag_hits(rag_hits: List[RagHit]) -> str:
    if not rag_hits:
        return ""
    excerpts = []
    for hit in rag_hits[:2]:
        text = hit.text.strip()
        if len(text) > 180:
            text = text[:180].rstrip() + "..."
        excerpts.append(f"第{hit.page}页：{text}")
    return "；".join(excerpts)
