from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from app.config import settings
from app.models.repair_unet import RepairUNet


class RepairService:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        image_size: int = 256
    ):
        self.output_dir = settings.output_dir / "repair"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else settings.repair_ckpt

        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            print(f"RepairService checkpoint not found: {self.checkpoint_path}")
            self.model = None
            return

        self.model = RepairUNet(in_channels=3, out_channels=3).to(self.device)
        state_dict = torch.load(str(self.checkpoint_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"RepairService loaded checkpoint: {self.checkpoint_path}")

    def is_available(self) -> bool:
        return self.model is not None

    def _preprocess(self, image: np.ndarray):
        """
        输入:
            RGB numpy, shape(H, W, 3)
        输出:
            tensor: [1, 3, image_size, image_size]
            original_size: (h, w)
        """
        if image is None:
            raise ValueError("输入图像为空")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"输入图像格式错误，期望 RGB 3通道，实际 shape={image.shape}")

        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.image_size, self.image_size))

        x = resized.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return x, (h, w)

    def _postprocess(self, output: torch.Tensor, original_size):
        """
        输出 tensor -> RGB uint8 numpy
        """
        output = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        h, w = original_size
        output = cv2.resize(output, (w, h))

        return output

    def _save_image(self, image: np.ndarray, path: Path):
        """
        保存 RGB 图像
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"保存修复结果失败: {path}")

    def repair(self, image: np.ndarray, stem: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        对输入 RGB 图像执行修复

        返回:
            repaired_path, aux_path
        其中 aux_path 当前保留为 None，方便以后扩展输出 mask / heatmap
        """
        if self.model is None:
            return None, None

        if image is None:
            return None, None

        if stem is None:
            stem = "repair_result"

        x, original_size = self._preprocess(image)

        with torch.no_grad():
            pred = self.model(x)

        repaired = self._postprocess(pred, original_size)

        repaired_path = self.output_dir / f"{stem}_repaired.png"
        self._save_image(repaired, repaired_path)

        return str(repaired_path.as_posix()), None


repair_service = RepairService()
