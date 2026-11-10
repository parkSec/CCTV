"""CLAHE: LAB 색공간의 L 채널에만 적용해서 색상은 유지한다."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer


class ClaheEnhancer(Enhancer):
    def __init__(self, clip_limit: float, tile_grid: int = 8):
        self.clip_limit = float(clip_limit)
        self.tile_grid = int(tile_grid)
        self.name = f"clahe_c{self.clip_limit}"
        self.params = {"clip_limit": self.clip_limit, "tile_grid": self.tile_grid}
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid))

    def enhance(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
