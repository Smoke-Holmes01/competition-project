import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from app.config import settings
from app.datasets import build_default_transform
from app.models.encoder import RetrievalEncoder
from app.services.metadata_service import metadata_service
from app.utils.mapping import group_to_obj_id, obj_id_to_glb_path, obj_id_to_glb_url, obj_id_to_path, obj_id_to_url


class PredictServiceUnavailable(RuntimeError):
    pass


class PredictService:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        raw_root: str | Path | None = None,
        image_size: int = 256,
        low_conf_threshold: float = 0.4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_conf_threshold = low_conf_threshold
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else settings.classifier_ckpt
        self.groups = self._load_groups()
        self.label_to_group = {i: group for i, group in enumerate(self.groups)}
        self.num_classes = len(self.groups)
        self.transform = build_default_transform(image_size, train=False)
        self.model: RetrievalEncoder | None = None
        self.unavailable_reason = ""
        self._load_model()

    def _load_groups(self) -> list[str]:
        if settings.mapping_path.exists():
            with settings.mapping_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                groups = [group for group in payload.keys() if group.startswith("group_")]
                if groups:
                    return sorted(groups, key=group_to_obj_id)
        return [f"group_{index:03d}" for index in range(1, 47)]

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            self.unavailable_reason = f"分类模型不存在: {self.checkpoint_path}"
            print(f"PredictService unavailable: {self.unavailable_reason}")
            return

        try:
            model = RetrievalEncoder(num_classes=self.num_classes)
            model.load_state_dict(
                torch.load(str(self.checkpoint_path), map_location=self.device)
            )
            model.to(self.device)
            model.eval()
            self.model = model
            print(f"PredictService initialized, classes: {self.num_classes}")
        except Exception as exc:
            self.unavailable_reason = f"分类模型加载失败: {exc}"
            self.model = None
            print(f"PredictService unavailable: {self.unavailable_reason}")

    def is_available(self) -> bool:
        return self.model is not None

    def status(self) -> dict:
        return {
            "available": self.is_available(),
            "reason": self.unavailable_reason,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "classes": self.num_classes,
        }

    def _ensure_available(self) -> RetrievalEncoder:
        if self.model is None:
            raise PredictServiceUnavailable(self.unavailable_reason or "分类模型不可用")
        return self.model

    def _read_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法读取上传图片，请确认文件格式是否正确")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _predict_probs_from_image(self, image: np.ndarray) -> torch.Tensor:
        model = self._ensure_available()
        tensor = self.transform(image=image)["image"]
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)

        return probs

    def predict_single_from_bytes(self, image_bytes: bytes):
        image = self._read_image_from_bytes(image_bytes)
        probs = self._predict_probs_from_image(image)
        return self._format_result(probs, views_used=1)

    def predict_multi_from_bytes_list(self, image_bytes_list: List[bytes]):
        if len(image_bytes_list) == 0:
            raise ValueError("没有上传任何图片")

        prob_sum = None
        valid_count = 0

        for image_bytes in image_bytes_list:
            image = self._read_image_from_bytes(image_bytes)
            probs = self._predict_probs_from_image(image)

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs

            valid_count += 1

        if valid_count == 0:
            raise ValueError("没有可用图片用于识别")

        avg_probs = prob_sum / valid_count
        return self._format_result(avg_probs, views_used=valid_count)

    def _format_result(self, probs: torch.Tensor, views_used: int):
        pred = probs.argmax(dim=1).item()
        final_group = self.label_to_group[pred]
        confidence = float(probs[0, pred].item())

        obj_id = group_to_obj_id(final_group)
        obj_path = obj_id_to_path(obj_id)
        obj_url = obj_id_to_url(obj_id)
        glb_path = obj_id_to_glb_path(obj_id)
        glb_url = obj_id_to_glb_url(obj_id)
        metadata = metadata_service.get_model_metadata(obj_id)

        topk = torch.topk(probs, k=min(5, self.num_classes), dim=1)

        top5 = []
        for score, idx in zip(topk.values[0], topk.indices[0]):
            top5.append({
                "group": self.label_to_group[idx.item()],
                "score": float(score.item())
            })

        message = "识别成功"
        if confidence < self.low_conf_threshold:
            message = "识别置信度较低，建议用户补拍更多角度"

        return {
            "success": True,
            "final_group": final_group,
            "obj_id": obj_id,
            "obj_path": obj_path,
            "obj_url": obj_url,
            "glb_path": glb_path,
            "glb_url": glb_url,
            "model_name": metadata["model_name"],
            "dynasty": metadata["dynasty"],
            "description": metadata["description"],
            "knowledge_id": metadata["knowledge_id"],
            "metadata": metadata,
            "confidence": confidence,
            "views_used": views_used,
            "top5": top5,
            "message": message,
        }
