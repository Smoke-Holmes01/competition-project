import numpy as np
import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum().item()
    union = (pred + target - pred * target).sum().item()
    return (inter + eps) / (union + eps)


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum().item()
    return (2 * inter + eps) / (pred.sum().item() + target.sum().item() + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
