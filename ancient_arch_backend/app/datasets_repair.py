from pathlib import Path
import random
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RepairDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        groups: Optional[List[str]] = None,
        image_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if not self.root_dir.exists():
            raise ValueError(f"目录不存在: {root_dir}")

        # 收集 group 目录
        all_group_dirs = sorted([
            p for p in self.root_dir.iterdir()
            if p.is_dir() and p.name.startswith("group_")
        ])

        if groups is not None:
            group_set = set(groups)
            group_dirs = [p for p in all_group_dirs if p.name in group_set]
        else:
            group_dirs = all_group_dirs

        self.samples = []
        for group_dir in group_dirs:
            for img_path in sorted(group_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append({
                        "group": group_dir.name,
                        "path": img_path
                    })

        if len(self.samples) == 0:
            raise ValueError("没有找到任何可用图像")

        print(f"RepairDataset initialized: groups={len(group_dirs)}, images={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"读取图像失败: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    def _random_erase(self, image: np.ndarray) -> np.ndarray:
        """
        在完整图上随机制造残缺区域
        """
        damaged = image.copy()
        h, w, _ = damaged.shape

        num_holes = random.randint(1, 4)

        for _ in range(num_holes):
            mode = random.choice(["rect", "circle"])

            if mode == "rect":
                hole_w = random.randint(w // 8, w // 3)
                hole_h = random.randint(h // 8, h // 3)
                x1 = random.randint(0, max(0, w - hole_w))
                y1 = random.randint(0, max(0, h - hole_h))

                fill_mode = random.choice(["black", "white", "noise"])
                if fill_mode == "black":
                    damaged[y1:y1 + hole_h, x1:x1 + hole_w] = 0
                elif fill_mode == "white":
                    damaged[y1:y1 + hole_h, x1:x1 + hole_w] = 255
                else:
                    noise = np.random.randint(0, 256, (hole_h, hole_w, 3), dtype=np.uint8)
                    damaged[y1:y1 + hole_h, x1:x1 + hole_w] = noise

            else:
                radius = random.randint(min(h, w) // 12, min(h, w) // 6)
                cx = random.randint(radius, w - radius)
                cy = random.randint(radius, h - radius)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)

                fill_mode = random.choice(["black", "white"])
                if fill_mode == "black":
                    damaged[mask == 255] = 0
                else:
                    damaged[mask == 255] = 255

        return damaged

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._read_image(sample["path"])
        damaged = self._random_erase(image)

        gt = image.astype(np.float32) / 255.0
        inp = damaged.astype(np.float32) / 255.0

        gt = torch.from_numpy(gt).permute(2, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp, gt
