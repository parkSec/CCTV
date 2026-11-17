"""데이터셋 공통 형식."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    image_id: str
    image_path: Path
    boxes: np.ndarray   # (N, 4) float32, xyxy 픽셀 좌표
    labels: np.ndarray  # (N,) int64, 공통 클래스 id
    ref_path: Path | None = None  # 정렬된 낮 이미지 (정답 기반 화질 지표용, DarkDriving)


class DetectionDataset(ABC):
    name: str
    eval_classes: list[int]

    @abstractmethod
    def load(self) -> list[Sample]:
        ...


def make_sample(image_id: str, image_path: Path, boxes: list, labels: list, ref_path: Path | None = None) -> Sample:
    return Sample(
        image_id=image_id,
        image_path=image_path,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        labels=np.asarray(labels, dtype=np.int64).reshape(-1),
        ref_path=ref_path,
    )


def select_samples(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """image_id로 정렬한 뒤 시드 고정으로 n개를 뽑는다. 입력 순서와 무관하게 같은 결과."""
    ordered = sorted(samples, key=lambda s: s.image_id)
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)
