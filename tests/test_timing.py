import time

import pytest

from src.timing import summarize_ms, timed


def test_timed_returns_result_and_ms():
    result, ms = timed(lambda x: (time.sleep(0.01), x * 2)[1], 21)
    assert result == 42
    assert ms >= 9.0


def test_timed_sync_without_torch_is_safe():
    result, ms = timed(lambda: "ok", sync_cuda=True)
    assert result == "ok" and ms >= 0


def test_summarize_ms():
    s = summarize_ms([1, 2, 3, 10])
    assert s["mean"] == pytest.approx(4.0)
    assert s["median"] == pytest.approx(2.5)
    assert s["n"] == 4


def test_summarize_ms_empty():
    assert summarize_ms([]) == {"mean": 0.0, "median": 0.0, "p95": 0.0, "n": 0}


def test_summarize_ms_p95():
    import numpy as np
    vals = [1, 2, 3, 4, 100]
    assert summarize_ms(vals)["p95"] == pytest.approx(float(np.percentile(vals, 95)))


def test_gpu_peak_without_cuda_is_safe():
    from src.timing import gpu_peak_mb, reset_gpu_peak
    reset_gpu_peak()
    v = gpu_peak_mb()
    assert v is None or v >= 0


def test_gpu_allocated_without_cuda_is_safe():
    from src.timing import gpu_allocated_mb
    v = gpu_allocated_mb()
    assert v is None or v >= 0
