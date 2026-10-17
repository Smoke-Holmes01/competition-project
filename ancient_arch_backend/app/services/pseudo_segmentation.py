from __future__ import annotations

import cv2
import numpy as np


def generate_pseudo_mask(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    image = image_rgb.astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    dist_white = np.linalg.norm(image.astype(np.int16) - 255, axis=2)
    dist_black = np.linalg.norm(image.astype(np.int16), axis=2)

    mask = (
        (dist_white > 45)
        & (dist_black > 35)
        & ((saturation > 18) | ((value > 35) & (value < 230)))
    ).astype(np.uint8) * 255

    h, w = mask.shape
    kernel_size = max(3, int(min(h, w) * 0.008))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    min_area = max(64, int(h * w * 0.0005))
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled
