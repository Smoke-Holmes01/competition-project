from pathlib import Path
import cv2
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, save_path: Path):
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    alpha = (mask[:, :, 0] > 0).astype(np.uint8)[:, :, None]
    vis = image.copy()
    vis = np.where(alpha == 1, (0.6 * image + 0.4 * color).astype(np.uint8), vis)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
