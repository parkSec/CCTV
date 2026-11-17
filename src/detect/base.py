"""탐지기 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray   # (N, 4) float32 xyxy
    labels: np.ndarray  # (N,) int64 공통 클래스 id
    scores: np.ndarray  # (N,) float32
    speed: dict[str, float] = field(default_factory=dict)  # Ultralytics Results.speed (preprocess, inference, postprocess ms)


class Detector(ABC):
    uses_cuda: bool = False

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> Detections:
        ...
