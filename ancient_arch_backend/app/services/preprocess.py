from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image

from app.config import settings


OUTPUT_DIR = settings.output_dir


def save_upload_bytes(content: bytes, suffix: str = ".png") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def crop_by_mask(image: np.ndarray, mask: np.ndarray, margin: int = 10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin, image.shape[1])
    y2 = min(int(ys.max()) + margin, image.shape[0])
    return image[y1:y2, x1:x2]


def pil_to_rgb_array(img: Image.Image):
    return np.array(img.convert("RGB"))
