"""Public responses and validated internal records; no framework state leaks into the API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(BaseModel):
    kind: Literal["chat", "knowledge", "image", "visual", "preview"] = "chat"
    operations: list[Literal["match", "segment", "repair", "preview"]] = Field(default_factory=list)
    needs_knowledge: bool = False
    image_positions: list[int] = Field(default_factory=list)
    all_images: bool = False
    clarification: str | None = None


class Asset(BaseModel):
    id: str
    conversation_id: str
    batch_id: str
    filename: str
    mime_type: str
    source_tool: str = "upload"
    parent_id: str | None = None
    role: str = "original"

    @property
    def url(self) -> str:
        return f"/agent/assets/{self.id}"


class RagHit(BaseModel):
    chunk_id: str
    source: str
    page: int | None = Field(default=None, ge=1)
    score: float = 0
    text: str


class StatusEvent(BaseModel):
    event: str
    message: str
    tool: str | None = None


class AgentToolCall(BaseModel):
    tool: str
    success: bool
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class Artifact(BaseModel):
    id: str
    type: Literal["image", "model", "json"]
    label: str
    source_tool: str
    url: str | None = None
    preview_url: str | None = None
    download_url: str | None = None
    mime_type: str | None = None


class AgentChatResponse(BaseModel):
    conversation_id: str
    answer: str
    tool_called: bool = False
    tool_result: dict[str, Any] | None = None
    rag_hits: list[RagHit] = Field(default_factory=list)
    render_instruction: dict[str, Any] | None = None
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    ui_instruction: dict[str, Any] | None = None
    status_trace: list[StatusEvent] = Field(default_factory=list)
