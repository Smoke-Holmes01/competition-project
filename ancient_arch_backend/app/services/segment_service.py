from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

from app.config import settings
from app.models.unet import UNet
from app.services.pseudo_segmentation import generate_pseudo_mask
from app.services.preprocess import crop_by_mask, save_gray, save_rgb


class SegmentService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.device == "cuda" else "cpu")
        self.model = None
        if settings.seg_ckpt.exists():
            self.model = UNet(in_channels=3, out_channels=1)
            state = torch.load(settings.seg_ckpt, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()

    def is_model_available(self) -> bool:
        return self.model is not None

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.model is None:
            return generate_pseudo_mask(image_rgb)

        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (settings.image_size, settings.image_size)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def segment_and_save(self, image_rgb: np.ndarray, stem: str):
        mask = self.predict_mask(image_rgb)
        crop = crop_by_mask(image_rgb, mask)
        mask_path = settings.output_dir / f"{stem}_mask.png"
        crop_path = settings.output_dir / f"{stem}_crop.png"
        save_gray(mask, mask_path)
        save_rgb(crop, crop_path)
        method = "unet" if self.is_model_available() else "pseudo_mask"
        return mask_path, crop_path, mask, crop, method, self.is_model_available()


segment_service = SegmentService()
