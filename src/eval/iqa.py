"""화질 지표 7개 (pyiqa).

- 정답 기반(FR): psnr, ssim, lpips — 정렬된 낮 이미지가 있는 데이터셋(DarkDriving)만
- 정답 없음(NR): niqe, brisque, musiq, clipiqa — 모든 데이터셋
값이 NaN·inf이거나 계산이 실패하면 그 지표만 NaN으로 두고 나머지는 계속 계산한다.
"""
from __future__ import annotations

import math
import statistics

import cv2
import numpy as np


class IQAUnavailable(RuntimeError):
    """pyiqa를 쓸 수 없을 때."""


def finite_or_nan(v: float) -> float:
    v = float(v)
    return v if math.isfinite(v) else float("nan")


def summarize_iqa(values: list[float]) -> dict:
    """{"mean", "median", "n", "n_nan"}. NaN은 평균·중앙값에서 빼고 개수만 센다."""
    ok = [v for v in values if not math.isnan(v)]
    return {"mean": float(statistics.fmean(ok)) if ok else float("nan"),
            "median": float(statistics.median(ok)) if ok else float("nan"),
            "n": len(ok), "n_nan": len(values) - len(ok)}


def _device(device):
    import torch
    if device in (None, "cuda") or isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}" if isinstance(device, int) else "cuda")
        return torch.device("cpu")
    return torch.device(device)


class IQAEvaluator:
    def __init__(self, metrics_nr: list[str], metrics_fr: list[str], device=0):
        try:
            import pyiqa
            import torch
        except ImportError as e:
            raise IQAUnavailable(f"pyiqa를 불러올 수 없음 (pip install pyiqa): {e}") from e
        self._torch = torch
        self.device = _device(device)
        self.metrics_nr = list(metrics_nr)
        self.metrics_fr = list(metrics_fr)
        self._m = {name: pyiqa.create_metric(name, device=self.device, as_loss=False)
                   for name in self.metrics_nr + self.metrics_fr}
        self.lower_better: dict[str, bool] = {n: bool(m.lower_better) for n, m in self._m.items()}

    def _tensor(self, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _run(self, name: str, *xs) -> float:
        try:
            with self._torch.no_grad():
                return finite_or_nan(self._m[name](*xs).mean().item())
        except Exception:  # 아주 어둡거나 단색인 이미지에서 NIQE 등이 실패할 수 있음
            return float("nan")

    def score(self, img_bgr: np.ndarray, ref_bgr: np.ndarray | None = None) -> dict[str, float]:
        if ref_bgr is not None and ref_bgr.shape[:2] != img_bgr.shape[:2]:
            raise ValueError(f"정답 이미지 크기 {ref_bgr.shape[:2]} != 이미지 크기 {img_bgr.shape[:2]}")
        x = self._tensor(img_bgr)
        out = {n: self._run(n, x) for n in self.metrics_nr}
        if ref_bgr is not None:
            r = self._tensor(ref_bgr)
            out.update({n: self._run(n, x, r) for n in self.metrics_fr})
        return out
