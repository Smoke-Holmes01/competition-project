"""Settings construction reads configuration; importing this module allocates no resources."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", case_sensitive=False)

    deepseek_api_key: str = Field(default="", repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    siliconflow_api_key: str = Field(default="", repr=False)
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    backend_base_url: str = "http://127.0.0.1:8000"
    public_asset_base_url: str = ""
    metadata_path: Path = ROOT.parent / "ancient_arch_backend/data/model_metadata.json"
    metadata_overrides_path: Path = ROOT.parent / "ancient_arch_backend/data/metadata_overrides.json"
    data_dir: Path = ROOT / "storage/runtime"
    kb_dir: Path = ROOT / "storage/index"
    kb_seed: Path = ROOT / "storage/kb/chunks.json"
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_images: int = Field(default=5, ge=1, le=20)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1)
    max_tool_calls: int = Field(default=8, ge=1, le=32)
    turn_timeout_seconds: float = Field(default=240, gt=0)
    request_timeout_seconds: float = Field(default=60, gt=0)
    context_char_budget: int = Field(default=32000, ge=1000)
    output_tokens: int = Field(default=2048, ge=128)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=1000, ge=100, le=2000)
    chunk_overlap: int = Field(default=150, ge=0)

    @property
    def database(self) -> Path:
        return self.data_dir / "agent.sqlite3"
