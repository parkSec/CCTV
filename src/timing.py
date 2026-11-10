"""처리 시간 측정."""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable


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
        return {"mean": 0.0, "median": 0.0, "n": 0}
    return {"mean": float(statistics.fmean(values)), "median": float(statistics.median(values)), "n": len(values)}
