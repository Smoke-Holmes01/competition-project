from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Ancient Arch Backend"
    debug: bool = True

    auth_enabled: bool = Field(default=True, alias="AUTH_ENABLED")
    auth_username: str = Field(default="admin", alias="AUTH_USERNAME")
    auth_password_hash: str = Field(default="", alias="AUTH_PASSWORD_HASH")
    auth_secret: str = Field(default="change-this-secret", alias="AUTH_SECRET")
    auth_cookie_name: str = Field(default="ancient_agent_session", alias="AUTH_COOKIE_NAME")
    auth_session_ttl_hours: int = Field(default=12, alias="AUTH_SESSION_TTL_HOURS")
    auth_cookie_secure: bool = Field(default=False, alias="AUTH_COOKIE_SECURE")

    data_root: Path = BASE_DIR / "data"
    raw_root: Path = BASE_DIR / "data" / "raw"
    mask_root: Path = BASE_DIR / "data" / "masks"
    renders_root: Path = BASE_DIR / "data" / "renders"
    splits_dir: Path = BASE_DIR / "data" / "splits"
    mapping_path: Path = BASE_DIR / "data" / "mapping.json"
    model_metadata_path: Path = BASE_DIR / "data" / "model_metadata.json"
    knowledge_path: Path = BASE_DIR / "data" / "knowledge.json"
    knowledge_pdf_path: Path = BASE_DIR.parent / "知识库.pdf"
    model_assets_dir: Path = BASE_DIR / "models"
    glb_assets_dir: Path = BASE_DIR / "models_glb"
    output_dir: Path = BASE_DIR / "outputs"
    metadata_overrides_path: Path = BASE_DIR / "data" / "metadata_overrides.json"

    classifier_ckpt: Path = BASE_DIR / "checkpoints" / "classifier.pth"
    seg_ckpt: Path = BASE_DIR / "checkpoints" / "seg_best.pth"
    retrieval_ckpt: Path = BASE_DIR / "checkpoints" / "retrieval_best.pth"
    repair_ckpt: Path = BASE_DIR / "checkpoints" / "repair_unet.pth"
    feature_db_path: Path = BASE_DIR / "checkpoints" / "feature_db.npz"

    image_size: int = 256
    device: str = "cuda"
    topk: int = 3
    repair_enabled: bool = True
    public_asset_base_url: str = Field(default="", alias="PUBLIC_ASSET_BASE_URL")
    cors_allow_origins: str = Field(default="", alias="CORS_ALLOW_ORIGINS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def normalized_public_asset_base_url(self) -> str:
        return self.public_asset_base_url.strip().rstrip("/")

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
