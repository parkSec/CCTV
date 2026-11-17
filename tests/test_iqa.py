import importlib.util
import math

import numpy as np
import pytest

from src.eval.iqa import summarize_iqa

HAS_PYIQA = importlib.util.find_spec("pyiqa") is not None
needs_pyiqa = pytest.mark.skipif(not HAS_PYIQA, reason="pyiqa 필요 (pip install pyiqa)")


def test_summarize_iqa_ignores_nan():
    s = summarize_iqa([1.0, float("nan"), 3.0])
    assert s["mean"] == 2.0 and s["median"] == 2.0 and s["n"] == 2 and s["n_nan"] == 1


def test_summarize_iqa_all_nan():
    s = summarize_iqa([float("nan")])
    assert math.isnan(s["mean"]) and s["n"] == 0 and s["n_nan"] == 1


def test_finite_or_nan():
    from src.eval.iqa import finite_or_nan
    assert finite_or_nan(1.5) == 1.5
    assert math.isnan(finite_or_nan(float("inf")))
    assert math.isnan(finite_or_nan(float("nan")))


def test_without_pyiqa_raises_unavailable():
    if HAS_PYIQA:
        pytest.skip("pyiqa가 있는 환경")
    from src.eval.iqa import IQAEvaluator, IQAUnavailable
    with pytest.raises(IQAUnavailable):
        IQAEvaluator(["niqe"], ["psnr"], device="cpu")


@pytest.fixture(scope="module")
def iqa():
    from src.eval.iqa import IQAEvaluator
    return IQAEvaluator(["niqe", "brisque", "musiq", "clipiqa"], ["psnr", "ssim", "lpips"], device=None)


def _img(seed=0, size=(256, 256)):
    return np.random.default_rng(seed).integers(0, 255, (*size, 3), dtype=np.uint8)


@needs_pyiqa
def test_fr_identical_images(iqa):
    img = _img()
    s = iqa.score(img, img)
    assert s["psnr"] >= 40 or math.isnan(s["psnr"])
    assert s["ssim"] == pytest.approx(1.0, abs=1e-3)
    assert s["lpips"] == pytest.approx(0.0, abs=1e-3)


@needs_pyiqa
def test_nr_only_without_ref(iqa):
    s = iqa.score(_img())
    assert set(s) == {"niqe", "brisque", "musiq", "clipiqa"}


@needs_pyiqa
def test_black_image_no_crash(iqa):
    s = iqa.score(np.zeros((256, 256, 3), np.uint8))
    assert all(isinstance(v, float) for v in s.values())


@needs_pyiqa
def test_fr_size_mismatch_raises(iqa):
    with pytest.raises(ValueError):
        iqa.score(_img(size=(256, 256)), _img(size=(128, 256)))


@needs_pyiqa
def test_lower_better_directions(iqa):
    lb = iqa.lower_better
    assert lb["niqe"] and lb["brisque"] and lb["lpips"]
    assert not (lb["musiq"] or lb["clipiqa"] or lb["psnr"] or lb["ssim"])
