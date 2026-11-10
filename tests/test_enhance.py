import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.enhance.base import IdentityEnhancer
from src.enhance.clahe import ClaheEnhancer
from src.enhance.gamma import GammaEnhancer
from src.enhance.registry import build_enhancers


@pytest.fixture
def dark():
    rng = np.random.default_rng(0)
    return rng.integers(0, 60, size=(37, 53, 3), dtype=np.uint8)


@pytest.mark.parametrize("enh", [IdentityEnhancer(), GammaEnhancer(2.0), ClaheEnhancer(3.0)])
def test_output_shape_dtype(enh, dark):
    out = enh.enhance(dark)
    assert out.shape == dark.shape
    assert out.dtype == np.uint8


def test_gamma_brightens_dark_image(dark):
    assert GammaEnhancer(2.0).enhance(dark).mean() > dark.mean()


def test_gamma_one_is_identity(dark):
    np.testing.assert_array_equal(GammaEnhancer(1.0).enhance(dark), dark)


def test_clahe_keeps_shape_and_changes_dark_image(dark):
    out = ClaheEnhancer(3.0, tile_grid=8).enhance(dark)
    assert out.shape == dark.shape
    assert not np.array_equal(out, dark)


def test_names():
    assert IdentityEnhancer().name == "original"
    assert GammaEnhancer(1.5).name == "gamma_g1.5"
    assert ClaheEnhancer(2.0).name == "clahe_c2.0"


def test_build_enhancers_expands_sweep():
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    enhancers, skipped = build_enhancers(cfg["conditions"])
    assert [e.name for e in enhancers] == [
        "original", "gamma_g1.5", "gamma_g2.0", "gamma_g2.5", "clahe_c2.0", "clahe_c3.0", "clahe_c4.0"]
    assert [s["condition"] for s in skipped] == ["zerodce", "realesrgan"]
    assert all(s["reason"] for s in skipped)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        build_enhancers([{"type": "magic"}])
