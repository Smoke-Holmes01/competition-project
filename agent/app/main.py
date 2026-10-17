from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from app.agent import ReactAgent
from app.auth import AuthResponse, LoginRequest, get_session_manager, require_auth
from app.config import get_settings
from app.schemas import AgentChatResponse, HistoryMessage, StatusEvent, TtsRequest


settings = get_settings()
session_manager = get_session_manager(settings)
react_agent = ReactAgent(settings)
MODEL_PROVIDERS = {"api", "local"}


def require_current_user(request: Request):
    return require_auth(settings, request)

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    description="ReAct Agent API for ancient architecture RAG and visual tool orchestration.",
    version="1.0.0",
)

if settings.cors_allow_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "siliconflow_base_url_configured": bool(settings.siliconflow_base_url.strip()),
        "siliconflow_model_configured": bool(settings.siliconflow_model.strip()),
        "local_llm_configured": settings.local_llm_configured,
        "local_model_available": settings.local_model_available,
        "backend_base_url": settings.normalized_backend_base_url,
        "public_asset_base_url": settings.normalized_public_asset_base_url,
        "knowledge_base": react_agent.rag.status(),
    }


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response):
    user = session_manager.authenticate(payload.username.strip(), payload.password)
    session_manager.set_session_cookie(response, user["username"])
    return AuthResponse(authenticated=True, username=user["username"])


@app.post("/auth/logout")
def logout(response: Response):
    session_manager.clear_session_cookie(response)
    return {"success": True}


@app.get("/auth/me", response_model=AuthResponse)
def me(user=Depends(require_current_user)):
    return AuthResponse(authenticated=True, username=user["username"])


@app.get("/agent/models")
def agent_models(user=Depends(require_current_user)):
    return {
        "default_provider": "api",
        "models": [
            {
                "provider": "api",
                "name": settings.api_model_display_name,
                "configured": settings.llm_configured,
                "available": settings.llm_configured,
            },
            {
                "provider": "local",
                "name": settings.local_model_name,
                "configured": settings.local_llm_configured,
                "available": settings.local_model_available,
                "path": str(settings.local_model_path),
            },
        ],
    }


def parse_model_provider(model_provider: str) -> str:
    provider = model_provider.strip().lower() or "api"
    if provider not in MODEL_PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid model_provider; expected api or local")
    return provider


async def fetch_tts_audio_from_url(audio_url: str) -> Response:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            media_type = response.headers.get("content-type") or "audio/mpeg"
            return Response(content=response.content, media_type=media_type)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"TTS audio HTTP {exc.response.status_code}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS audio fetch failed: {exc}") from exc


def extract_tts_audio_url(payload: dict) -> Optional[str]:
    for key in ("audio_url", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    audio = payload.get("audio")
    if isinstance(audio, dict):
        value = audio.get("url") or audio.get("audio_url")
        if isinstance(value, str) and value.strip():
            return value
    return None


@app.post("/agent/tts")
async def tts(payload: TtsRequest, user=Depends(require_current_user)):
    if not settings.tts_configured or settings.tts_speech_url is None:
        raise HTTPException(status_code=503, detail="TTS 未配置：请设置 TTS_MODEL，并按需设置 TTS_BASE_URL/TTS_API_KEY")

    voice = settings.tts_voice.strip() if payload.voice == "boy" else payload.voice.strip()
    request_payload = {
        "model": settings.tts_model.strip(),
        "input": payload.text.strip(),
        "voice": voice or "boy",
        "response_format": settings.tts_response_format.strip() or "mp3",
        "speed": payload.rate,
    }
    headers = {"Content-Type": "application/json"}
    api_key = settings.resolved_tts_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(settings.tts_speech_url, headers=headers, json=request_payload)
            response.raise_for_status()
            media_type = response.headers.get("content-type") or "audio/mpeg"
            if "application/json" in media_type:
                audio_url = extract_tts_audio_url(response.json())
                if audio_url:
                    return await fetch_tts_audio_from_url(audio_url)
                raise HTTPException(status_code=502, detail="TTS 返回 JSON 但没有音频地址")
            return Response(content=response.content, media_type=media_type)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"TTS HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS 调用失败: {exc}") from exc


def parse_history_payload(history: str) -> list[HistoryMessage]:
    if not history.strip():
        return []
    try:
        payload = json.loads(history)
        if not isinstance(payload, list):
            raise ValueError("history must be a list")
        return [HistoryMessage.model_validate(item) for item in payload]
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid history payload: {exc}") from exc


async def read_upload_images(
    *,
    image: Optional[UploadFile],
    images: Optional[list[UploadFile]],
) -> list[dict]:
    uploads: list[UploadFile] = []
    if images:
        uploads.extend(images)
    if image is not None:
        uploads.append(image)

    payload: list[dict] = []
    for index, upload in enumerate(uploads):
        image_bytes = await upload.read()
        if not image_bytes:
            continue
        payload.append(
            {
                "bytes": image_bytes,
                "filename": upload.filename or f"upload_{index + 1}.png",
                "content_type": upload.content_type or "application/octet-stream",
            }
        )
    return payload


@app.post("/agent/chat", response_model=AgentChatResponse)
async def chat(
    message: str = Form(...),
    conversation_id: str = Form(default=""),
    history: str = Form(default="[]"),
    model_provider: str = Form(default="api"),
    image: Optional[UploadFile] = File(default=None),
    images: Optional[list[UploadFile]] = File(default=None),
    user=Depends(require_current_user),
):
    image_payload = await read_upload_images(image=image, images=images)
    return await react_agent.run(
        message=message,
        conversation_id=conversation_id,
        history=parse_history_payload(history),
        images=image_payload,
        model_provider=parse_model_provider(model_provider),
    )


@app.post("/agent/stream")
async def stream(
    message: str = Form(...),
    conversation_id: str = Form(default=""),
    history: str = Form(default="[]"),
    model_provider: str = Form(default="api"),
    image: Optional[UploadFile] = File(default=None),
    images: Optional[list[UploadFile]] = File(default=None),
    user=Depends(require_current_user),
):
    parsed_model_provider = parse_model_provider(model_provider)
    image_payload = await read_upload_images(image=image, images=images)
    history_messages = parse_history_payload(history)

    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def sink(status: StatusEvent) -> None:
            if status.event == "final":
                return
            await queue.put(
                {
                    "event": status.event,
                    "data": json.dumps(status.model_dump(), ensure_ascii=False),
                }
            )

        async def answer_sink(content: str) -> None:
            if not content:
                return
            await queue.put(
                {
                    "event": "answer_delta",
                    "data": json.dumps({"content": content}, ensure_ascii=False),
                }
            )

        async def runner():
            try:
                result = await react_agent.run(
                    message=message,
                    conversation_id=conversation_id,
                    history=history_messages,
                    images=image_payload,
                    model_provider=parsed_model_provider,
                    status_sink=sink,
                    answer_sink=answer_sink,
                )
                await queue.put(
                    {
                        "event": "final",
                        "data": result.model_dump_json(),
                    }
                )
            except Exception as exc:
                await queue.put(
                    {
                        "event": "error",
                        "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
                    }
                )
            finally:
                await queue.put({"event": "done", "data": "[DONE]"})

        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                yield item
                if item["event"] == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(event_generator())
