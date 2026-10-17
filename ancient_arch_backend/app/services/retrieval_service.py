from pathlib import Path
import uuid
import numpy as np
import torch
import cv2

from app.config import settings
from app.models.encoder import RetrievalEncoder


class RetrievalService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.device == "cuda" else "cpu")
        self.model = RetrievalEncoder(backbone_name="resnet18", embedding_dim=256, pretrained=False)
        self.available = False
        if settings.retrieval_ckpt.exists():
            state = torch.load(settings.retrieval_ckpt, map_location=self.device)
            self.model.load_state_dict(state)
            self.available = True
        self.model.to(self.device).eval()
        self.db = None
        self.obj_ids = None
        if settings.feature_db_path.exists():
            data = np.load(settings.feature_db_path, allow_pickle=True)
            self.db = data["features"].astype(np.float32)
            self.obj_ids = data["obj_ids"].astype(np.int32)

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image_rgb, (settings.image_size, settings.image_size)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return tensor

    def encode(self, image_rgb: np.ndarray) -> np.ndarray:
        x = self._preprocess(image_rgb)
        with torch.no_grad():
            emb = self.model(x)[0].cpu().numpy().astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb

    def search(self, image_rgb: np.ndarray, topk: int = 3):
        if self.db is None or self.obj_ids is None:
            raise RuntimeError("特征库不存在，请先运行 scripts/build_feature_db.py")
        query = self.encode(image_rgb)
        db = self.db / (np.linalg.norm(self.db, axis=1, keepdims=True) + 1e-8)
        scores = db @ query
        idxs = np.argsort(-scores)[:topk]
        results = []
        for idx in idxs:
            results.append({"obj_id": int(self.obj_ids[idx]), "score": float(scores[idx])})
        return results

    def vote_multi(self, images: list[np.ndarray], topk: int = 3):
        agg = {}
        for img in images:
            for item in self.search(img, topk=topk):
                agg.setdefault(item["obj_id"], []).append(item["score"])
        final = [{"obj_id": k, "score": float(np.mean(v))} for k, v in agg.items()]
        final = sorted(final, key=lambda x: x["score"], reverse=True)
        return final[:topk]


retrieval_service = RetrievalService()
