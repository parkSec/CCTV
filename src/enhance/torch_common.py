"""PyTorch 개선 모델 래퍼 공통 부분. torch가 없어도 import할 수 있다."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Enhancer, EnhancerUnavailable

try:
    import torch
except ImportError:  # GPU 없는 작업 환경
    torch = None


def require_torch(what: str) -> None:
    if torch is None:
        raise EnhancerUnavailable(f"torch가 설치되어 있지 않아 {what}을(를) 쓸 수 없음")


def require_file(path: Path, hint: str = "") -> Path:
    path = Path(path)
    if not path.exists():
        raise EnhancerUnavailable(f"파일 없음: {path}" + (f" ({hint})" if hint else ""))
    return path


def check_variant(variant: str, allowed: tuple[str, ...], model: str) -> str:
    if variant not in allowed:
        raise ValueError(f"{model} variant {variant!r}는 없음 (가능: {list(allowed)})")
    return variant


def pick_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(path: Path, device):
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class TorchRGBEnhancer(Enhancer):
    """입력 RGB [0,1] (1,3,H,W) 텐서, 출력도 같은 형태인 모델의 공통 준비·후처리.
    하위 클래스는 self.net, self.device를 만들고 infer만 정의한다."""

    device = None

    def _set_device(self, device) -> None:
        self.device = pick_device(device)
        self.uses_cuda = self.device.type == "cuda"

    def preprocess(self, img: np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def postprocess(self, y, img_shape: tuple[int, int]) -> np.ndarray:
        h, w = img_shape
        y = (y.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)[:h, :w]
