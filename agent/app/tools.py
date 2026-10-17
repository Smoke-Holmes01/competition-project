from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.schemas import RenderInstruction, ToolResult


class VisionToolClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def segment_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "upload.png",
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            data = await self._post_visual_endpoint(
                client,
                "/segment",
                image_bytes,
                filename=filename,
                content_type=content_type,
                action_label="图像分割",
            )
        return self._normalize_asset_payload(data)

    async def repair_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "upload.png",
        content_type: str = "application/octet-stream",
        segmented: bool = False,
    ) -> Dict[str, Any]:
        endpoint = "/repair/segmented" if segmented else "/repair"
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            data = await self._post_visual_endpoint(
                client,
                endpoint,
                image_bytes,
                filename=filename,
                content_type=content_type,
                action_label="图像修复",
            )
        return self._normalize_asset_payload(data)

    async def predict_single_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "upload.png",
        content_type: str = "application/octet-stream",
    ) -> ToolResult:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            predict_data = await self._post_visual_endpoint(
                client,
                "/predict/single",
                image_bytes,
                filename=filename,
                content_type=content_type,
                action_label="视觉匹配",
            )
        if not predict_data.get("ok"):
            return ToolResult(
                success=False,
                error=predict_data.get("error") or "视觉匹配失败",
                status_code=predict_data.get("status_code"),
            )
        return await self._tool_result_from_predict_payload(predict_data.get("payload") or {})

    async def predict_multi_images(
        self,
        images: List[Dict[str, Any]],
    ) -> ToolResult:
        url = f"{self.settings.normalized_backend_base_url}/predict/multi"
        files = [
            (
                "files",
                (
                    image.get("filename") or f"upload_{index + 1}.png",
                    image["bytes"],
                    image.get("content_type") or "application/octet-stream",
                ),
            )
            for index, image in enumerate(images)
        ]
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, files=files)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                error=f"多图融合匹配接口 HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"多图融合匹配接口不可用: {exc}")

        return await self._tool_result_from_predict_payload(payload)

    def preview_model(self, obj_id: int) -> ToolResult:
        return ToolResult(
            success=True,
            obj_id=obj_id,
            final_group=f"group_{obj_id:03d}",
            obj_path=f"models/{obj_id}.obj",
            obj_url=self._public_asset_url(f"/models/{obj_id}.obj"),
            glb_path=f"models_glb/{obj_id}.glb",
            glb_url=self._public_asset_url(f"/models_glb/{obj_id}.glb"),
        )

    async def match_uploaded_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "upload.png",
        content_type: str = "application/octet-stream",
    ) -> ToolResult:
        preprocessing: Dict[str, Any] = {
            "original_filename": filename,
            "segmentation": None,
            "repair": None,
            "predict_input": "original",
            "fallback_used": False,
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            segment_data = await self._post_visual_endpoint(
                client,
                "/segment",
                image_bytes,
                filename=filename,
                content_type=content_type,
                action_label="图像分割",
            )
            if not segment_data.get("ok"):
                return ToolResult(
                    success=False,
                    preprocessing=preprocessing,
                    error=segment_data.get("error") or "图像分割失败",
                    status_code=segment_data.get("status_code"),
                )

            preprocessing["segmentation"] = segment_data.get("payload")
            segment_payload = segment_data.get("payload") or {}
            crop_url = segment_payload.get("crop_url")
            crop_bytes = await self._download_asset_bytes(client, crop_url) if crop_url else None

            predict_bytes = crop_bytes or image_bytes
            predict_filename = self._derive_filename(filename, suffix="_crop") if crop_bytes else filename
            preprocessing["predict_input"] = "segmented_crop" if crop_bytes else "original"
            preprocessing["fallback_used"] = crop_bytes is None
            if crop_url and crop_bytes is None:
                preprocessing["segmentation_asset_warning"] = "分割成功，但裁剪图无法下载，已回退到原图匹配"

            repair_data = await self._post_visual_endpoint(
                client,
                "/repair/segmented",
                image_bytes,
                filename=filename,
                content_type=content_type,
                action_label="图像修复",
                allow_soft_failure=True,
            )
            if repair_data.get("payload"):
                preprocessing["repair"] = repair_data.get("payload")
                repaired_url = (repair_data.get("payload") or {}).get("repaired_url")
                repaired_bytes = await self._download_asset_bytes(client, repaired_url) if repaired_url else None
                if repaired_bytes:
                    predict_bytes = repaired_bytes
                    predict_filename = self._derive_filename(filename, suffix="_repaired")
                    preprocessing["predict_input"] = "repaired_segmented"
                    preprocessing["fallback_used"] = False
                elif repaired_url:
                    preprocessing["repair_asset_warning"] = "修复成功，但修复图无法下载，已继续使用分割图/原图匹配"
            elif repair_data.get("error"):
                preprocessing["repair"] = {
                    "success": False,
                    "message": repair_data.get("error"),
                    "soft_failure": True,
                }

            predict_data = await self._post_visual_endpoint(
                client,
                "/predict/single",
                predict_bytes,
                filename=predict_filename,
                content_type=content_type,
                action_label="视觉匹配",
            )
            if not predict_data.get("ok"):
                return ToolResult(
                    success=False,
                    preprocessing=preprocessing,
                    error=predict_data.get("error") or "视觉匹配失败",
                    status_code=predict_data.get("status_code"),
                )

            predict_payload = predict_data.get("payload") or {}

        result = await self._tool_result_from_predict_payload(predict_payload)
        result.preprocessing = preprocessing
        return result

    async def _tool_result_from_predict_payload(self, predict_payload: Dict[str, Any]) -> ToolResult:
        obj_id = predict_payload.get("obj_id")

        return ToolResult(
            success=bool(predict_payload.get("success", True)),
            obj_id=obj_id,
            final_group=predict_payload.get("final_group"),
            obj_path=predict_payload.get("obj_path"),
            obj_url=self._public_asset_url(predict_payload.get("obj_url")),
            glb_path=predict_payload.get("glb_path"),
            glb_url=self._public_asset_url(predict_payload.get("glb_url")),
            model_name=predict_payload.get("model_name"),
            dynasty=predict_payload.get("dynasty"),
            description=predict_payload.get("description"),
            knowledge_id=predict_payload.get("knowledge_id"),
            metadata=predict_payload.get("metadata"),
            confidence=predict_payload.get("confidence"),
            views_used=predict_payload.get("views_used"),
            top5=predict_payload.get("top5") or [],
            error=None if predict_payload.get("success", True) else predict_payload.get("message", "视觉匹配失败"),
        )

    async def _post_visual_endpoint(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        action_label: str,
        allow_soft_failure: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.settings.normalized_backend_base_url}{endpoint}"
        files = {"file": (filename, image_bytes, content_type)}
        try:
            response = await client.post(url, files=files)
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "payload": payload, "status_code": response.status_code}
        except httpx.HTTPStatusError as exc:
            error = f"{action_label}接口 HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            if allow_soft_failure:
                return {"ok": False, "payload": None, "status_code": exc.response.status_code, "error": error}
            return {"ok": False, "payload": None, "status_code": exc.response.status_code, "error": error}
        except Exception as exc:
            error = f"{action_label}接口不可用: {exc}"
            if allow_soft_failure:
                return {"ok": False, "payload": None, "status_code": None, "error": error}
            return {"ok": False, "payload": None, "status_code": None, "error": error}

    async def _download_asset_bytes(self, client: httpx.AsyncClient, value: Optional[str]) -> Optional[bytes]:
        asset_url = self._backend_asset_url(value)
        if not asset_url:
            return None
        try:
            response = await client.get(asset_url)
            response.raise_for_status()
            return response.content
        except Exception:
            return None

    def _normalize_asset_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data.get("payload") or {})
        for key, value in list(payload.items()):
            if key.endswith("_url") and isinstance(value, str):
                payload[key] = self._public_asset_url(value)
        normalized = dict(data)
        normalized["payload"] = payload
        return normalized

    def _backend_asset_url(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if value.startswith("/"):
            return f"{self.settings.normalized_backend_base_url}{value}"
        return f"{self.settings.normalized_backend_base_url}/{value}"

    def _public_asset_url(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value

        public_base = self.settings.normalized_public_asset_base_url
        if value.startswith("http://") or value.startswith("https://"):
            parsed = urlsplit(value)
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            if public_base:
                return f"{public_base}{path}"
            if value.startswith(self.settings.normalized_backend_base_url):
                return path
            return value
        elif not value.startswith("/"):
            path = f"/{value}"
        else:
            path = value

        if public_base:
            return f"{public_base}{path}"
        return path

    def _derive_filename(self, filename: str, *, suffix: str) -> str:
        if "." not in filename:
            return f"{filename}{suffix}"
        stem, ext = filename.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"


def build_render_instruction(tool_result: Optional[ToolResult]) -> Optional[RenderInstruction]:
    if not tool_result or not tool_result.success or tool_result.obj_id is None:
        return None

    model_path = tool_result.glb_path or tool_result.obj_path
    model_url = tool_result.glb_url or tool_result.obj_url
    model_format = "glb" if (tool_result.glb_url or tool_result.glb_path) else "obj"

    return RenderInstruction(
        model_id=int(tool_result.obj_id),
        model_path=model_path,
        model_url=model_url,
        model_format=model_format,
        confidence=tool_result.confidence,
    )
