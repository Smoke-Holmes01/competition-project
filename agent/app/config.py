from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Ancient Arch ReAct Agent"
    debug: bool = True

    auth_enabled: bool = Field(default=True, alias="AUTH_ENABLED")
    auth_username: str = Field(default="admin", alias="AUTH_USERNAME")
    auth_password_hash: str = Field(default="", alias="AUTH_PASSWORD_HASH")
    auth_secret: str = Field(default="change-this-secret", alias="AUTH_SECRET")
    auth_cookie_name: str = Field(default="ancient_agent_session", alias="AUTH_COOKIE_NAME")
    auth_session_ttl_hours: int = Field(default=12, alias="AUTH_SESSION_TTL_HOURS")
    auth_cookie_secure: bool = Field(default=False, alias="AUTH_COOKIE_SECURE")

    siliconflow_base_url: str = Field(default="", alias="SILICONFLOW_BASE_URL")
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    siliconflow_model: str = Field(default="", alias="SILICONFLOW_MODEL")
    api_model_display_name: str = Field(default="Qwen3.5-397B-A17B", alias="API_MODEL_DISPLAY_NAME")
    tts_base_url: str = Field(default="", alias="TTS_BASE_URL")
    tts_api_key: str = Field(default="", alias="TTS_API_KEY")
    tts_model: str = Field(default="", alias="TTS_MODEL")
    tts_voice: str = Field(default="boy", alias="TTS_VOICE")
    tts_response_format: str = Field(default="mp3", alias="TTS_RESPONSE_FORMAT")

    local_model_path: Path = Field(default=Path("/home/yy/models/Qwen3.5-9B"), alias="LOCAL_MODEL_PATH")
    local_model_name: str = Field(default="Qwen3.5-9B", alias="LOCAL_MODEL_NAME")

    backend_base_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_BASE_URL")
    public_asset_base_url: str = Field(default="", alias="PUBLIC_ASSET_BASE_URL")
    cors_allow_origins: str = Field(default="", alias="CORS_ALLOW_ORIGINS")
    kb_pdf_path: Path = Field(default=BASE_DIR.parent / "知识库.pdf", alias="KB_PDF_PATH")
    kb_cache_dir: Path = Field(default=BASE_DIR / "storage" / "kb", alias="KB_CACHE_DIR")

    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_min_score: float = Field(default=0.02, alias="RAG_MIN_SCORE")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def llm_configured(self) -> bool:
        return bool(self.siliconflow_base_url.strip() and self.siliconflow_model.strip())

    @property
    def local_llm_configured(self) -> bool:
        return bool(str(self.local_model_path).strip())

    @property
    def local_model_available(self) -> bool:
        return self.local_model_path.exists()

    @property
    def chat_completions_url(self) -> Optional[str]:
        base_url = self.siliconflow_base_url.strip().rstrip("/")
        if not base_url:
            return None
        return f"{base_url}/chat/completions"

    @property
    def models_url(self) -> Optional[str]:
        base_url = self.siliconflow_base_url.strip().rstrip("/")
        if not base_url:
            return None
        return f"{base_url}/models"

    @property
    def tts_speech_url(self) -> Optional[str]:
        base_url = (self.tts_base_url.strip() or self.siliconflow_base_url.strip()).rstrip("/")
        if not base_url:
            return None
        return f"{base_url}/audio/speech"

    @property
    def tts_configured(self) -> bool:
        return bool(self.tts_speech_url and self.tts_model.strip())

    @property
    def resolved_tts_api_key(self) -> str:
        return self.tts_api_key.strip() or self.siliconflow_api_key.strip()

    @property
    def normalized_backend_base_url(self) -> str:
        return self.backend_base_url.strip().rstrip("/")

    @property
    def normalized_public_asset_base_url(self) -> str:
        return self.public_asset_base_url.strip().rstrip("/")

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def resolved_kb_pdf_path(self) -> Path:
        path = self.kb_pdf_path
        if path.is_absolute():
            return path
        return (BASE_DIR / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.kb_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
