import asyncio
import logging
import re
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage, message_to_dict, messages_from_dict

from app.assets import inspect_image
from app.errors import AgentError
from app.graph import RunContext
from app.schemas import AgentChatResponse, AgentToolCall, Artifact, Asset, RagHit

logger = logging.getLogger(__name__)


def recent_history(groups: list[list[dict]], budget: int):
    chosen, total = [], 0
    for group in reversed(groups):
        messages = messages_from_dict(group)
        cost = sum(
            len(str(message.content)) + len(str(getattr(message, "tool_calls", []))) for message in messages
        )
        if total + cost > budget:
            break
        total += cost
        chosen.insert(0, messages)
    return [message for group in chosen for message in group]


def project(state: dict, trace) -> AgentChatResponse:
    calls = [AgentToolCall.model_validate(row) for row in state["calls"]]
    artifacts = {}
    tool_result = render = None
    target = "knowledge" if state["rag_hits"] else None
    for call in calls:
        if not call.success:
            continue
        result = call.result or {}
        for record in result.get("assets", []):
            asset = Asset.model_validate(record)
            label = {"crop": "分割主体", "mask": "分割掩膜", "repaired": "修复图片"}.get(
                asset.role, asset.role
            )
            artifacts[asset.id] = Artifact(
                id=asset.id,
                type="image",
                label=label,
                source_tool=call.tool,
                url=asset.url,
                preview_url=asset.url,
                download_url=asset.url,
                mime_type=asset.mime_type,
            )
            target = "recognition"
        if "obj_id" in result:
            tool_result = {k: v for k, v in result.items() if k != "render_instruction"}
            render = result.get("render_instruction")
            target = "model" if render else "recognition"
            if render:
                render = {**render, "confidence": result.get("confidence")}
                model_id, url = result["obj_id"], render["model_url"]
                artifacts[f"model_{model_id}"] = Artifact(
                    id=f"model_{model_id}",
                    type="model",
                    label=result.get("model_name") or f"模型 {model_id}",
                    source_tool=call.tool,
                    url=url,
                    download_url=url,
                )
    return AgentChatResponse(
        conversation_id=state["conversation_id"],
        answer=state["answer"],
        tool_called=bool(calls),
        tool_result=tool_result,
        render_instruction=render,
        tool_calls=calls,
        rag_hits=[RagHit.model_validate(row) for row in state["rag_hits"]],
        artifacts=list(artifacts.values()),
        ui_instruction={"target_view": target} if target else None,
        status_trace=trace,
    )


class AgentService:
    def __init__(self, settings, store, assets, agent_graph, checkpointer):
        self.settings, self.store, self.assets = settings, store, assets
        self.agent_graph, self.checkpointer = agent_graph, checkpointer
        self.busy: dict[str, object] = {}

    def reserve(self, conversation_id: str) -> str:
        conversation_id = conversation_id.strip() or "conv_" + uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", conversation_id):
            raise AgentError("会话编号格式不正确。")
        if conversation_id in self.busy:
            raise AgentError("该会话正在处理请求，请等待完成。", status=409)
        self.busy[conversation_id] = object()
        return conversation_id

    def release(self, conversation_id: str, lease: object):
        # An old connection's cleanup must never unlock a newer turn.
        if self.busy.get(conversation_id) is lease:
            self.busy.pop(conversation_id, None)

    async def run(self, conversation_id: str, message: str, uploads: list[tuple[bytes, str]], sink):
        turn_id = None
        lease = self.busy.get(conversation_id)
        started = time.monotonic()
        context = RunContext(sink=sink)
        try:
            async with asyncio.timeout(self.settings.turn_timeout_seconds):
                if not message.strip():
                    raise AgentError("消息不能为空。")
                if len(uploads) > self.settings.max_images:
                    raise AgentError(f"每次最多上传 {self.settings.max_images} 张图片。")
                # Validate the whole batch before persisting any image.
                for data, _ in uploads:
                    await asyncio.to_thread(inspect_image, data, self.settings)
                batch_id = uuid4().hex
                new_assets = [
                    await self.assets.save(conversation_id, batch_id, data, name) for data, name in uploads
                ]
                available = await self.store.latest_assets(conversation_id)
                turn_id = await self.store.begin(conversation_id, message, new_assets)
                history = recent_history(
                    await self.store.history_groups(conversation_id), self.settings.context_char_budget
                )
                state = {
                    "conversation_id": conversation_id,
                    "user_message": message,
                    "messages": [*history, HumanMessage(content=message)],
                    "assets": [asset.model_dump() for asset in available],
                    "intent": {},
                    "calls": [],
                    "rag_hits": [],
                    "failures": {},
                    "steps": 0,
                    "stop_reason": None,
                    "answer": "",
                }
                result = await self.agent_graph.graph.ainvoke(
                    state,
                    {
                        "configurable": {"thread_id": conversation_id},
                        "recursion_limit": self.settings.max_tool_calls * 4 + 16,
                    },
                    context=context,
                )
                response = project(result, context.trace)
                await self.store.finish(
                    turn_id,
                    response.model_dump(),
                    [message_to_dict(m) for m in result["messages"][len(history) :]],
                )
                logger.info(
                    "agent_turn_complete conversation=%s tools=%d elapsed_seconds=%.3f",
                    conversation_id,
                    result["steps"],
                    time.monotonic() - started,
                )
                return response
        except asyncio.CancelledError:
            if turn_id:
                await asyncio.shield(
                    self.store.fail(turn_id, "cancelled", "请求已取消，未完成的任务没有继续执行。")
                )
            raise
        except Exception as exc:
            if isinstance(exc, AgentError):
                error = exc
            elif isinstance(exc, TimeoutError):
                error = AgentError("本轮请求超过总时限，已停止后续执行。", status=504)
            else:
                status = getattr(exc, "status_code", None)
                message = (
                    f"模型 API 返回 HTTP {status}，请检查配置或稍后重试。"
                    if status
                    else "请求处理失败，请检查模型连接或服务端日志后重试。"
                )
                error = AgentError(message, status=502)
                logger.warning(
                    "agent_turn_failed conversation=%s kind=%s", conversation_id, type(exc).__name__
                )
            if turn_id:
                await self.store.fail(turn_id, "failed", str(error))
            raise error from exc
        finally:
            self.release(conversation_id, lease)

    async def delete(self, conversation_id: str):
        self.reserve(conversation_id)
        lease = self.busy[conversation_id]
        try:
            await self.checkpointer.adelete_thread(conversation_id)
            ids = await self.store.delete_conversation(conversation_id)
            await self.assets.delete_files(ids)
        finally:
            self.release(conversation_id, lease)
