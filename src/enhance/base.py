"""화질 개선 기법 공통 인터페이스. 입출력은 BGR uint8, 출력 크기 == 입력 크기.

처리 시간을 공정하게 비교하려고 세 단계로 나눈다.
- preprocess: 패딩, 색 변환, 텐서 변환, GPU 전송
- infer: 모델 계산 (전통 기법은 계산 전부)
- postprocess: 자르기, GPU→CPU, uint8·BGR 변환
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.timing import cuda_sync


class EnhancerUnavailable(RuntimeError):
    """가중치·라이브러리가 없어 이 기법을 쓸 수 없을 때."""


class Enhancer(ABC):
    name: str = ""
    params: dict = {}
    uses_cuda: bool = False

    @abstractmethod
    def preprocess(self, img: np.ndarray) -> Any:
        ...

    @abstractmethod
    def infer(self, x: Any) -> Any:
        ...

    @abstractmethod
    def postprocess(self, y: Any, img_shape: tuple[int, int]) -> np.ndarray:
        ...

    def enhance(self, img: np.ndarray) -> np.ndarray:
        return self.postprocess(self.infer(self.preprocess(img)), img.shape[:2])

    def enhance_timed(self, img: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """(결과, {pre_ms, infer_ms, post_ms, total_ms}). GPU를 쓰면 단계마다 동기화해서 잰다."""
        sync = cuda_sync if self.uses_cuda else (lambda: None)
        sync()
        t0 = time.perf_counter()
        x = self.preprocess(img)
        sync()
        t1 = time.perf_counter()
        y = self.infer(x)
        sync()
        t2 = time.perf_counter()
        out = self.postprocess(y, img.shape[:2])
        sync()
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000.0
        return out, {"pre_ms": ms(t0, t1), "infer_ms": ms(t1, t2), "post_ms": ms(t2, t3), "total_ms": ms(t0, t3)}


class PassThroughEnhancer(Enhancer):
    """전통 기법용: 준비·후처리는 그대로 넘기고 계산은 infer에서만 한다."""

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return img

    def postprocess(self, y: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
        return y


class IdentityEnhancer(PassThroughEnhancer):
    name = "original"

    def __init__(self):
        self.params = {}

    def infer(self, x: np.ndarray) -> np.ndarray:
        return x
