"""긴 변 제한 리사이즈. 모든 데이터셋 이미지를 같은 상한(설정 preprocess.max_side)으로 맞춘다."""
from __future__ import annotations

import cv2
import numpy as np


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """(줄인 이미지, 배율). 긴 변이 max_side 이하이면 원본 그대로, 배율 1.0 (확대하지 않음)."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img, 1.0
    s = max_side / long_side
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), s


def scale_boxes(boxes: np.ndarray, s: float) -> np.ndarray:
    """xyxy 박스에 배율을 곱한다."""
    return (np.asarray(boxes, np.float32).reshape(-1, 4) * np.float32(s)).astype(np.float32)
