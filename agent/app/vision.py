"""Only HTTP integration lives here. No visual inference code is imported."""

import json
from urllib.parse import urlsplit

import httpx

from app.assets import Assets
from app.config import Settings
from app.errors import AgentError
from app.schemas import Asset


class Vision:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, assets: Assets):
        self.settings, self.client, self.assets = settings, client, assets
        self.catalog = {}
        for path in (settings.metadata_path, settings.metadata_overrides_path):
            if path.is_file():
                records = json.loads(path.read_text(encoding="utf-8"))
                for key, value in records.items():
                    key = str(value.get("model_id", key))
                    self.catalog[key] = {**self.catalog.get(key, {}), **value}

    def internal_url(self, path: str, prefixes=("/outputs/",)) -> str:
        parsed = urlsplit(path)
        base = urlsplit(self.settings.backend_base_url)
        if (
            (parsed.scheme and parsed.scheme not in {"http", "https"})
            or (parsed.netloc and (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc))
            or not any(parsed.path.startswith(prefix) for prefix in prefixes)
        ):
            raise AgentError("视觉服务返回了不允许访问的资源地址。")
        if ".." in parsed.path or "%" in parsed.path or "\\" in parsed.path:
            raise AgentError("视觉服务返回了无效资源路径。")
        return self.settings.backend_base_url.rstrip("/") + parsed.path

    def public_url(self, path: str) -> str:
        return self.settings.public_asset_base_url.rstrip("/") + urlsplit(path).path

    async def post(self, endpoint: str, inputs: list[Asset], multi=False) -> dict:
        files = [
            ("files" if multi else "file", (a.filename, await self.assets.read(a), a.mime_type))
            for a in inputs
        ]
        try:
            response = await self.client.post(
                self.settings.backend_base_url.rstrip("/") + endpoint, files=files
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("success") is not True:
                raise AgentError("视觉服务未完成操作，请检查所需权重和资源。")
            return payload
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise AgentError(
                f"视觉接口返回 HTTP {status}。", retryable=status == 429 or status >= 500, status=502
            ) from exc
        except httpx.RequestError as exc:
            raise AgentError("视觉服务暂时无法连接或请求超时。", retryable=True, status=502) from exc
        except (ValueError, TypeError) as exc:
            raise AgentError("视觉接口返回了无效结果。", status=502) from exc

    async def download(self, path: str) -> bytes:
        url = self.internal_url(path)
        try:
            async with self.client.stream("GET", url) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for data in response.aiter_bytes():
                    size += len(data)
                    if size > self.settings.max_image_bytes:
                        raise AgentError("工具图片产物超过大小限制。")
                    chunks.append(data)
                return b"".join(chunks)
        except httpx.HTTPStatusError as exc:
            raise AgentError("工具图片产物不可访问。", retryable=exc.response.status_code >= 500) from exc
        except httpx.RequestError as exc:
            raise AgentError("下载工具产物超时或连接失败。", retryable=True) from exc

    async def process(self, operation: str, asset: Asset) -> dict:
        payload = await self.post("/segment" if operation == "segment_image" else "/repair", [asset])
        fields = (
            [("crop_url", "crop"), ("mask_url", "mask")]
            if operation == "segment_image"
            else [("repaired_url", "repaired")]
        )
        results = []
        for field, role in fields:
            if not payload.get(field):
                raise AgentError("视觉接口没有返回完整图片产物。")
            data = await self.download(payload[field])
            derived = await self.assets.save(
                asset.conversation_id,
                asset.batch_id,
                data,
                f"{role}.png",
                source_tool=operation,
                parent_id=asset.id,
                role=role,
            )
            results.append(derived.model_dump())
        return {
            "assets": results,
            "method": payload.get("method"),
            "model_available": payload.get("model_available"),
        }

    async def resource(self, model_id: int) -> dict | None:
        for directory, extension in (("models_glb", "glb"), ("models", "obj")):
            path = f"/{directory}/{model_id}.{extension}"
            try:
                response = await self.client.head(self.settings.backend_base_url.rstrip("/") + path)
                if response.status_code == 405:
                    async with self.client.stream(
                        "GET", self.settings.backend_base_url.rstrip("/") + path
                    ) as r:
                        exists = r.is_success
                else:
                    exists = response.is_success
                if exists:
                    return {
                        "type": "model",
                        "model_id": model_id,
                        "model_url": self.public_url(path),
                        "model_format": extension,
                    }
            except httpx.RequestError as exc:
                raise AgentError("无法检查三维模型资源，视觉服务连接失败。", retryable=True) from exc
        return None

    async def preview(self, model_id: int) -> dict:
        if not self.catalog:
            raise AgentError("模型元数据缺失，无法校验预览编号。")
        if str(model_id) not in self.catalog:
            raise AgentError("该模型编号不在已有模型目录内，请核实编号。")
        render = await self.resource(model_id)
        if not render:
            raise AgentError(f"模型 {model_id} 的 GLB/OBJ 资源不可访问。")
        item = self.catalog[str(model_id)]
        return {
            "obj_id": model_id,
            "model_name": item.get("model_name"),
            "dynasty": item.get("dynasty"),
            "description": item.get("description"),
            "render_instruction": render,
        }

    async def match(self, inputs: list[Asset], multi: bool) -> dict:
        payload = await self.post("/predict/multi" if multi else "/predict/single", inputs, multi)
        if not isinstance(payload.get("obj_id"), int):
            raise AgentError("视觉匹配没有返回有效模型编号。")
        result = {
            key: payload.get(key)
            for key in (
                "obj_id",
                "final_group",
                "model_name",
                "dynasty",
                "description",
                "confidence",
                "views_used",
                "top5",
                "knowledge_id",
                "metadata",
            )
        }
        try:
            result["render_instruction"] = await self.resource(payload["obj_id"])
        except AgentError:
            result["render_instruction"] = None
        if not result["render_instruction"]:
            result["resource_warning"] = "匹配已返回候选，但三维资源暂不可访问。"
        return result
