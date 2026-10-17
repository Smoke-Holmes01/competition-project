from pathlib import Path
from typing import List
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from app.utils.io import read_lines


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ======================
# transforms
# ======================
def build_default_transform(image_size: int, train: bool = True):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()
    ])


def build_seg_transform(image_size: int, train: bool = True):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()
    ])


# ======================
# utils
# ======================
def list_group_images(group_dir: Path) -> List[Path]:
    if not group_dir.exists():
        return []
    return sorted([p for p in group_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


# ======================
# ⭐ 分类Dataset：直接读取 raw 下全部 group
# ======================
class ClassificationDataset(Dataset):
    def __init__(self, raw_root: Path, image_size=256, train=True):
        self.raw_root = Path(raw_root)

        self.groups = sorted([p.name for p in self.raw_root.iterdir() if p.is_dir()])
        self.group_to_label = {g: i for i, g in enumerate(self.groups)}
        self.label_to_group = {i: g for i, g in enumerate(self.groups)}

        self.samples = []
        for g in self.groups:
            for img_path in list_group_images(self.raw_root / g):
                self.samples.append((img_path, self.group_to_label[g]))

        self.transform = build_default_transform(image_size, train)

        print("Classification classes:", len(self.groups))
        print("Classification samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        return image, label


# ======================
# segmentation dataset（保留）
# ======================
class SegmentationDataset(Dataset):
    def __init__(self, raw_root: Path, mask_root: Path, split_file: Path, image_size=256, train=True):
        self.groups = read_lines(split_file)
        self.raw_root = Path(raw_root)
        self.mask_root = Path(mask_root)

        self.samples = []
        for g in self.groups:
            for img_path in list_group_images(self.raw_root / g):
                mask_path = self.mask_root / g / img_path.name
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        self.transform = build_seg_transform(image_size, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        aug = self.transform(image=image, mask=mask)

        image = aug["image"]
        mask = aug["mask"].float().unsqueeze(0) / 255.0

        return image, mask, str(img_path)


class TripletGroupDataset(Dataset):
    def __init__(self, data_root: Path, split_file: Path, renders_root: Path, image_size=256):
        self.groups = read_lines(split_file)
        self.raw_root = Path(data_root)
        self.renders_root = Path(renders_root)
        self.transform = build_default_transform(image_size, train=True)

        self.raw_by_group = {}
        self.render_by_group = {}
        for group in self.groups:
            obj_id = int(group.split("_")[-1])
            raw_images = list_group_images(self.raw_root / group)
            render_images = list_group_images(self.renders_root / str(obj_id))
            if raw_images and render_images:
                self.raw_by_group[group] = raw_images
                self.render_by_group[group] = render_images

        self.groups = [group for group in self.groups if group in self.raw_by_group]
        self.samples = [(group, img_path) for group in self.groups for img_path in self.raw_by_group[group]]

        if len(self.groups) < 2:
            raise ValueError("TripletGroupDataset requires at least two groups with raw and render images")

    def __len__(self):
        return len(self.samples)

    def _load_tensor(self, path: Path):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]

    def __getitem__(self, idx):
        group, anchor_path = self.samples[idx]
        positive_path = random.choice(self.render_by_group[group])
        negative_group = random.choice([g for g in self.groups if g != group])
        negative_path = random.choice(self.render_by_group[negative_group])
        return self._load_tensor(anchor_path), self._load_tensor(positive_path), self._load_tensor(negative_path)
