"""FastAPI transport. Both chat endpoints execute the same application service."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from app.assets import Assets
from app.config import Settings
from app.errors import AgentError
from app.graph import AgentGraph
from app.knowledge import Knowledge
from app.providers import DeepSeekModel, Embeddings
from app.schemas import AgentChatResponse
from app.service import AgentService
from app.storage import open_store
from app.toolbox import Toolbox
from app.vision import Vision


def create_app(
    settings: Settings | None = None,
    *,
    model=None,
    embeddings=None,
    vision_factory=Vision,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.data_dir.mkdir(parents=True, exist_ok=True)
        store = await open_store(config.database)
        try:
            async with (
                httpx.AsyncClient(
                    timeout=config.request_timeout_seconds, transport=transport, follow_redirects=False
                ) as client,
                AsyncSqliteSaver.from_conn_string(str(config.database)) as checkpointer,
            ):
                await checkpointer.setup()
                assets = Assets(store, config)
                llm = model or DeepSeekModel(config)
                embedder = embeddings or Embeddings(config, client)
                knowledge = Knowledge(config, embedder)
                vision = vision_factory(config, client, assets)
                toolbox = Toolbox(assets, knowledge, vision)
                graph = AgentGraph(config, llm, toolbox, assets, checkpointer)
                app.state.service = AgentService(config, store, assets, graph, checkpointer)
                app.state.settings, app.state.knowledge = config, knowledge
                try:
                    yield
                finally:
                    close = getattr(llm, "aclose", None)
                    if close:
                        await close()
        finally:
            await store.db.close()

    api = FastAPI(title="Ancient Architecture Agent", version="2.0.0", lifespan=lifespan)
    origins = config.cors_allow_origins
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @api.exception_handler(AgentError)
    async def expected_error(request: Request, exc: AgentError):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @api.get("/health")
    async def health():
        config = api.state.settings
        return {
            "status": "ok",
            "version": "2.0.0",
            "model": config.deepseek_model,
            "deepseek_configured": bool(config.deepseek_api_key),
            "embedding_configured": bool(config.siliconflow_api_key),
            "knowledge_ready": api.state.knowledge.vectors is not None,
            "knowledge_error": api.state.knowledge.error,
        }

    @api.get("/agent/models")
    async def models():
        config = api.state.settings
        return {
            "default_provider": "api",
            "models": [
                {
                    "provider": "api",
                    "name": "DeepSeek V4.1 Flash",
                    "model": config.deepseek_model,
                    "configured": bool(config.deepseek_api_key),
                }
            ],
        }

    @api.get("/agent/conversations/{conversation_id}")
    async def conversation(conversation_id: str):
        result = await api.state.service.store.conversation(conversation_id)
        if result is None:
            raise HTTPException(404, "会话不存在。")
        return result

    @api.delete("/agent/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str):
        await api.state.service.delete(conversation_id)
        return Response(status_code=204)

    @api.get("/agent/assets/{asset_id}")
    async def asset_file(asset_id: str):
        service = api.state.service
        asset = await service.store.asset(asset_id)
        if not asset or not service.assets.path(asset).is_file():
            raise HTTPException(404, "图片不存在。")
        return FileResponse(
            service.assets.path(asset), media_type=asset.mime_type, headers={"Cache-Control": "no-store"}
        )

    async def prepare(conversation_id, model_provider, image, images):
        if model_provider not in {"", "api"}:
            raise AgentError("仅支持 DeepSeek API，本地模型已移除。")
        uploads = [*(images or []), *([image] if image else [])]
        config = api.state.settings
        if len(uploads) > config.max_images:
            raise AgentError(f"每次最多上传 {config.max_images} 张图片。")
        payload = []
        for upload in uploads:
            data = await upload.read(config.max_image_bytes + 1)
            payload.append((data, upload.filename or "upload.png"))
        return api.state.service.reserve(conversation_id), payload

    @api.post("/agent/chat", response_model=AgentChatResponse)
    async def chat(
        request: Request,
        message: Annotated[str, Form(min_length=1, max_length=8000)],
        conversation_id: Annotated[str, Form()] = "",
        history: Annotated[str, Form()] = "[]",
        model_provider: Annotated[str, Form()] = "api",
        image: Annotated[UploadFile | None, File()] = None,
        images: Annotated[list[UploadFile] | None, File()] = None,
    ):
        conversation_id, uploads = await prepare(conversation_id, model_provider, image, images)
        lease = api.state.service.busy[conversation_id]

        async def discard(event, data):
            pass

        started = False

        async def run():
            nonlocal started
            started = True
            return await api.state.service.run(conversation_id, message, uploads, discard)

        task = asyncio.create_task(run())

        async def disconnected():
            # The form has already been consumed. Listen directly: is_disconnected()
            # uses a cancellation scope that can swallow this watcher's cancellation.
            while True:
                event = await request.receive()
                if event["type"] == "http.disconnect":
                    return

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return await task
            raise AgentError("客户端已断开，请求已取消。", status=499)
        finally:
            for pending in (task, watcher):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, watcher, return_exceptions=True)
            if not started:
                api.state.service.release(conversation_id, lease)

    @api.post("/agent/stream")
    async def stream(
        request: Request,
        message: Annotated[str, Form(min_length=1, max_length=8000)],
        conversation_id: Annotated[str, Form()] = "",
        history: Annotated[str, Form()] = "[]",
        model_provider: Annotated[str, Form()] = "api",
        image: Annotated[UploadFile | None, File()] = None,
        images: Annotated[list[UploadFile] | None, File()] = None,
    ):
        conversation_id, uploads = await prepare(conversation_id, model_provider, image, images)
        lease = api.state.service.busy[conversation_id]
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        started = False

        async def sink(event, data):
            await queue.put({"event": event, "data": json.dumps(data, ensure_ascii=False)})

        async def runner():
            nonlocal started
            started = True
            try:
                result = await api.state.service.run(conversation_id, message, uploads, sink)
                await sink("final", result.model_dump())
            except AgentError as exc:
                await sink("error", {"error": str(exc), "conversation_id": conversation_id})

        async def events():
            task = asyncio.create_task(runner())
            try:
                yield {"event": "session", "data": json.dumps({"conversation_id": conversation_id})}
                while not task.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.2)
                    except TimeoutError:
                        continue
                    yield event
                yield {"event": "done", "data": "[DONE]"}
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if not started:
                    api.state.service.release(conversation_id, lease)

        async def cleanup():
            # Also covers disconnect before the generator's first iteration.
            api.state.service.release(conversation_id, lease)

        return EventSourceResponse(events(), background=BackgroundTask(cleanup))

    return api


app = create_app()
