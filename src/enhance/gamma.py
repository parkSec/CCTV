"""Gamma 보정: out = 255 * (in/255)^(1/gamma). gamma > 1 이면 밝아진다."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer


class GammaEnhancer(Enhancer):
    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError("gamma는 0보다 커야 합니다")
        self.gamma = float(gamma)
        self.name = f"gamma_g{self.gamma}"
        self.params = {"gamma": self.gamma}
        x = np.arange(256, dtype=np.float64) / 255.0
        self._lut = np.clip(np.round(255.0 * x ** (1.0 / self.gamma)), 0, 255).astype(np.uint8)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        return cv2.LUT(img, self._lut)
