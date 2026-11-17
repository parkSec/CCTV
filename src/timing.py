"""처리 시간 측정."""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import numpy as np


def cuda_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable, *args, sync_cuda: bool = False) -> tuple[Any, float]:
    """fn(*args)를 실행하고 (결과, 밀리초)를 돌려준다. GPU 연산은 sync_cuda=True로 비동기 실행까지 기다린다."""
    if sync_cuda:
        cuda_sync()
    t0 = time.perf_counter()
    result = fn(*args)
    if sync_cuda:
        cuda_sync()
    return result, (time.perf_counter() - t0) * 1000.0


def summarize_ms(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "n": 0}
    return {"mean": float(statistics.fmean(values)), "median": float(statistics.median(values)),
            "p95": float(np.percentile(values, 95)), "n": len(values)}


def _cuda_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def reset_gpu_peak() -> None:
    """조건마다 GPU 최대 메모리 기록을 0으로 되돌린다. CUDA가 없으면 아무것도 안 함."""
    torch = _cuda_torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float | None:
    """마지막 reset_gpu_peak 이후 PyTorch가 잡은 GPU 메모리 최대치(MB). CUDA가 없으면 None."""
    torch = _cuda_torch()
    if torch is None:
        return None
    return torch.cuda.max_memory_allocated() / 2**20
