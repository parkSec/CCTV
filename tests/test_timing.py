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
    assert summarize_ms([]) == {"mean": 0.0, "median": 0.0, "n": 0}
