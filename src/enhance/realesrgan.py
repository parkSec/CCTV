"""Real-ESRGAN (xinntao/Real-ESRGAN 공식 realesrgan 패키지). 업스케일 후 원래 해상도로 되돌려 탐지기 입력 크기를 맞춘다.

- 공식 패키지가 의존하는 basicsr는 Python 3.13+에서 설치되지 않으므로 Python 3.11/3.12 환경이 필요하다.
- torchvision 0.17+에는 torchvision.transforms.functional_tensor가 없어 basicsr import가 실패하므로 호환 패치를 먼저 적용한다.
- GPU 메모리가 부족하면 타일 크기를 절반으로 줄여 다시 시도하고, 줄인 크기를 다음 이미지에도 계속 쓴다.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .base import Enhancer, EnhancerUnavailable


def patch_torchvision_functional_tensor() -> bool:
    """functional_tensor가 없으면 functional을 그 이름으로 등록한다. 패치했으면 True."""
    name = "torchvision.transforms.functional_tensor"
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        pass
    try:
        functional = importlib.import_module("torchvision.transforms.functional")
    except ImportError:
        return False
    sys.modules[name] = functional
    return True


def run_with_tile_retry(fn: Callable[[int], Any], tile: int, min_tile: int,
                        is_oom: Callable[[BaseException], bool],
                        on_oom: Callable[[], None] | None = None) -> tuple[Any, int]:
    """fn(tile)을 실행. 메모리 부족이면 tile을 절반으로 줄여 재시도. (결과, 성공한 tile)."""
    while True:
        try:
            return fn(tile), tile
        except Exception as e:
            if not is_oom(e):
                raise
            if on_oom:
                on_oom()
            if tile // 2 < min_tile:
                raise
            tile //= 2


def _is_cuda_oom(e: BaseException) -> bool:
    import torch
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


class RealESRGANEnhancer(Enhancer):
    def __init__(self, weights: Path, scale: int = 2, tile: int = 400, min_tile: int = 128,
                 half: bool = True, device: str | None = None):
        weights = Path(weights)
        if not weights.exists():
            raise EnhancerUnavailable(f"가중치 파일 없음: {weights}")
        try:
            import torch
        except ImportError as e:
            raise EnhancerUnavailable("torch가 설치되어 있지 않아 realesrgan을 쓸 수 없음") from e
        patch_torchvision_functional_tensor()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as e:
            raise EnhancerUnavailable(f"realesrgan/basicsr 패키지를 불러올 수 없음 (Python 3.11/3.12 필요): {e}") from e

        use_cuda = torch.cuda.is_available() and device != "cpu"
        self._torch = torch
        self.scale = int(scale)
        self.tile = int(tile)
        self.min_tile = int(min_tile)
        self.uses_cuda = use_cuda
        self.name = f"realesrgan_x{self.scale}"
        self.params = {"scale": self.scale, "tile": self.tile, "half": bool(half and use_cuda), "weights": weights.name}
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
        self.upsampler = RealESRGANer(
            scale=self.scale, model_path=str(weights), model=model, tile=self.tile, tile_pad=10, pre_pad=0,
            half=bool(half and use_cuda), device=torch.device("cuda" if use_cuda else "cpu"))

    def _run(self, img: np.ndarray, tile: int) -> np.ndarray:
        self.upsampler.tile_size = tile
        # 공식 구현이 타일마다 'Tile i/n'을 print하므로 숨긴다
        with contextlib.redirect_stdout(io.StringIO()):
            out, _ = self.upsampler.enhance(img, outscale=self.scale)
        return out

    def enhance(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        on_oom = self._torch.cuda.empty_cache if self.uses_cuda else None
        out, self.tile = run_with_tile_retry(lambda t: self._run(img, t), self.tile, self.min_tile,
                                             _is_cuda_oom, on_oom)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
