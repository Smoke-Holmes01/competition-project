from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StatusEvent(BaseModel):
    event: str
    message: str


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str = "boy"
    rate: float = Field(default=1.0, ge=0.5, le=1.5)


class RagHit(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    text: str


class ToolResult(BaseModel):
    success: bool
    obj_id: Optional[int] = None
    final_group: Optional[str] = None
    obj_path: Optional[str] = None
    obj_url: Optional[str] = None
    glb_path: Optional[str] = None
    glb_url: Optional[str] = None
    model_name: Optional[str] = None
    dynasty: Optional[str] = None
    description: Optional[str] = None
    knowledge_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    views_used: Optional[int] = None
    top5: List[Dict[str, Any]] = Field(default_factory=list)
    preprocessing: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class AgentToolCall(BaseModel):
    tool: str
    success: bool
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentArtifact(BaseModel):
    id: str
    type: Literal["image", "model", "json"]
    label: str
    url: Optional[str] = None
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    mime_type: Optional[str] = None
    source_tool: str


class UiInstruction(BaseModel):
    target_view: Literal["recognition", "model", "knowledge"]
    active_artifact_id: Optional[str] = None
    message: Optional[str] = None


class RenderInstruction(BaseModel):
    type: str = "model"
    model_id: int
    model_path: Optional[str] = None
    model_url: Optional[str] = None
    model_format: str = "obj"
    confidence: Optional[float] = None


class AgentChatResponse(BaseModel):
    conversation_id: str
    answer: str
    tool_called: bool
    tool_result: Optional[ToolResult] = None
    rag_hits: List[RagHit] = Field(default_factory=list)
    render_instruction: Optional[RenderInstruction] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    artifacts: List[AgentArtifact] = Field(default_factory=list)
    ui_instruction: Optional[UiInstruction] = None
    status_trace: List[StatusEvent] = Field(default_factory=list)
