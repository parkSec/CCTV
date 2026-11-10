"""화질 개선 기법 공통 인터페이스. 입출력은 BGR uint8, 출력 크기 == 입력 크기."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EnhancerUnavailable(RuntimeError):
    """가중치·라이브러리가 없어 이 기법을 쓸 수 없을 때."""


class Enhancer(ABC):
    name: str = ""
    params: dict = {}
    uses_cuda: bool = False

    @abstractmethod
    def enhance(self, img: np.ndarray) -> np.ndarray:
        ...


class IdentityEnhancer(Enhancer):
    name = "original"

    def __init__(self):
        self.params = {}

    def enhance(self, img: np.ndarray) -> np.ndarray:
        return img
