import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.enhance.base import IdentityEnhancer
from src.enhance.clahe import ClaheEnhancer
from src.enhance.gamma import GammaEnhancer
from src.enhance.registry import build_enhancers
from tests.enhancer_contract import assert_enhancer_contract


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
    enhancers, skipped = build_enhancers([{"type": "gamma", "sweep": {"gamma": [1.5, 2.0]}}])
    assert [e.name for e in enhancers] == ["gamma_g1.5", "gamma_g2.0"]
    assert skipped == []


def test_build_default_conditions(tmp_path):
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    # 가중치가 PC에 있든 없든 같은 결과가 나오도록, 딥러닝 기법은 없는 경로를 가리키게 한다
    for c in cfg["conditions"]:
        if "weights" in c:
            c["weights"] = tmp_path / "missing.pth"
    # third_party 저장소를 받은 PC에서도 같은 결과가 나오도록 빈 폴더를 가리키게 한다
    enhancers, skipped = build_enhancers(cfg["conditions"], third_party_root=tmp_path / "third_party")
    assert [e.name for e in enhancers] == ["original", "gamma_g1.5", "clahe_c2.0"]
    assert [s["condition"] for s in skipped] == ["zerodce", "zerodcepp", "sci", "ruas", "nightenh"]
    assert all(s["reason"] for s in skipped)


def test_unknown_type_skipped():
    enhancers, skipped = build_enhancers([{"type": "magic"}])
    assert enhancers == []
    assert skipped[0]["condition"] == "magic" and "magic" in skipped[0]["reason"]


@pytest.mark.parametrize("enh", [IdentityEnhancer(), GammaEnhancer(1.5), ClaheEnhancer(2.0, 8)])
def test_contract(enh):
    assert_enhancer_contract(enh)


@pytest.mark.parametrize("enh", [IdentityEnhancer(), GammaEnhancer(1.5), ClaheEnhancer(2.0, 8)])
def test_enhance_equals_staged(enh, dark):
    staged = enh.postprocess(enh.infer(enh.preprocess(dark)), dark.shape[:2])
    np.testing.assert_array_equal(enh.enhance(dark), staged)


def test_enhance_timed_total_covers_stages(dark):
    _, t = ClaheEnhancer(2.0, 8).enhance_timed(dark)
    assert t["total_ms"] >= t["infer_ms"]
