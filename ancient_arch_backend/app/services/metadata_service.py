from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


class MetadataService:
    def __init__(self, metadata_path: Path | None = None):
        self.metadata_path = metadata_path or settings.model_metadata_path
        self.overrides_path = settings.metadata_overrides_path
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        if not self.metadata_path.exists():
            return self._apply_overrides(metadata)

        with self.metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metadata = {str(item.get("model_id", key)): item for key, item in data.items()}
        return self._apply_overrides(metadata)

    def _apply_overrides(self, metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not self.overrides_path.exists():
            return metadata

        with self.overrides_path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            return metadata

        for key, override in overrides.items():
            if not isinstance(override, dict):
                continue
            item = dict(metadata.get(str(key), {}))
            item.update(override)
            metadata[str(key)] = item
        return metadata

    def get_model_metadata(self, obj_id: int) -> dict[str, Any]:
        item = self._metadata.get(str(obj_id), {})
        fallback = {
            "model_id": obj_id,
            "group": f"group_{obj_id:03d}",
            "knowledge_id": f"knowledge_{obj_id:03d}",
            "model_name": f"斗拱模型 {obj_id}",
            "dynasty": "待补充",
            "description": "知识库元数据尚未生成，请运行 scripts/extract_knowledge_pdf.py。",
            "source": "fallback",
        }
        fallback.update(item)
        return fallback


metadata_service = MetadataService()
