"""전통 기법 6개(HE, AGCWD, MSRCR, LIME, Dong, BIMEF) 테스트."""
import numpy as np
import pytest

from src.enhance.registry import build_enhancers
from src.enhance.traditional import (
    AgcwdEnhancer,
    BimefEnhancer,
    DongEnhancer,
    HistEqEnhancer,
    LimeEnhancer,
    MsrcrEnhancer,
    agcwd_lut,
    bimef_btf,
    refine_illumination,
)
from tests.enhancer_contract import assert_enhancer_contract


def _dark_scene(h=90, w=160, seed=0):
    """어두운 도로 비슷한 장면: 어두운 바탕 + 밝은 광원 하나 + 중간 밝기 사각형(차)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(5, 40, (h, w, 3), dtype=np.uint8)
    img[20:30, 10:20] = 250                # 헤드라이트
    img[50:80, 60:120] = rng.integers(30, 70, (30, 60, 3), dtype=np.uint8)  # 차
    return img


ALL = [
    HistEqEnhancer(),
    AgcwdEnhancer(),
    MsrcrEnhancer(),
    LimeEnhancer(),
    DongEnhancer(),
    BimefEnhancer(),
]


@pytest.mark.parametrize("enh", ALL, ids=lambda e: e.name)
def test_contract(enh):
    assert_enhancer_contract(enh)


@pytest.mark.parametrize("enh", ALL, ids=lambda e: e.name)
def test_brightens_dark_scene(enh):
    img = _dark_scene()
    assert enh.enhance(img).mean() > img.mean() + 5


@pytest.mark.parametrize("enh", ALL, ids=lambda e: e.name)
def test_deterministic(enh):
    img = _dark_scene()
    np.testing.assert_array_equal(enh.enhance(img), enh.enhance(img))


def test_names():
    assert [e.name for e in ALL] == ["he", "agcwd_a0.5", "msrcr", "lime", "dong", "bimef"]


def test_he_spreads_luminance():
    img = _dark_scene()
    out = HistEqEnhancer().enhance(img)
    # 평활화 후 밝기 분포가 0~255 전체로 퍼진다
    assert np.percentile(out, 99) - np.percentile(out, 1) > np.percentile(img, 99) - np.percentile(img, 1)


def test_agcwd_lut_monotonic_and_fixes_ends():
    v = _dark_scene()[..., 0]
    lut = agcwd_lut(v, alpha=0.5)
    assert lut.dtype == np.uint8 and lut.shape == (256,)
    assert np.all(np.diff(lut.astype(int)) >= 0)
    assert lut[0] == 0 and lut[255] == 255


def test_agcwd_dark_image_gets_curve_above_identity():
    v = _dark_scene()[..., 0]
    lut = agcwd_lut(v, alpha=0.5)
    assert np.all(lut[1:255].astype(int) >= np.arange(1, 255))


def test_agcwd_flat_histogram_does_not_crash():
    v = np.full((10, 10), 7, dtype=np.uint8)
    lut = agcwd_lut(v, alpha=0.5)
    assert np.all(np.diff(lut.astype(int)) >= 0)


def test_refine_illumination_is_smoother_than_input_and_keeps_level():
    rng = np.random.default_rng(1)
    t_hat = np.clip(0.3 + 0.1 * rng.standard_normal((60, 80)), 0.01, 1)
    t = refine_illumination(t_hat, alpha=0.5, weight="lime")
    assert t.shape == t_hat.shape
    assert np.abs(np.diff(t, axis=1)).mean() < np.abs(np.diff(t_hat, axis=1)).mean()
    assert abs(t.mean() - t_hat.mean()) < 0.02


def test_refine_illumination_alpha_zero_is_identity():
    t_hat = np.random.default_rng(2).random((20, 30))
    np.testing.assert_allclose(refine_illumination(t_hat, alpha=0.0, weight="lime"), t_hat, atol=1e-6)


def test_bimef_btf_k1_is_identity_and_k_gt1_brightens():
    p = np.linspace(0.01, 1, 50)
    np.testing.assert_allclose(bimef_btf(p, 1.0), p, atol=1e-9)
    assert np.all(bimef_btf(p[:-1], 4.0) > p[:-1])


def test_dong_keeps_bright_light_bright():
    img = _dark_scene()
    out = DongEnhancer().enhance(img)
    assert out[20:30, 10:20].mean() > 200


def test_build_traditional_conditions():
    conds = [
        {"type": "he"},
        {"type": "agcwd", "alpha": 0.5},
        {"type": "msrcr"},
        {"type": "lime"},
        {"type": "dong"},
        {"type": "bimef"},
    ]
    enhancers, skipped = build_enhancers(conds)
    assert [e.name for e in enhancers] == ["he", "agcwd_a0.5", "msrcr", "lime", "dong", "bimef"]
    assert skipped == []


def test_fast_blur_close_to_exact_for_large_sigma():
    from src.enhance.traditional import _blur
    import cv2
    img = (_dark_scene(240, 400).astype(np.float32) + 1.0)
    exact = cv2.GaussianBlur(img, (0, 0), 80)
    approx = _blur(img, 80)
    assert approx.shape == exact.shape
    assert np.abs(approx - exact).max() / exact.mean() < 0.03


def test_fast_blur_small_sigma_is_exact():
    from src.enhance.traditional import _blur
    import cv2
    img = _dark_scene().astype(np.float32)
    np.testing.assert_allclose(_blur(img, 15), cv2.GaussianBlur(img, (0, 0), 15), rtol=1e-6)


def test_guided_filter_keeps_constant_and_shape():
    from src.enhance.traditional import _guided_filter
    guide = _dark_scene()[..., 0].astype(np.float64) / 255
    flat = np.full(guide.shape, 0.4)
    out = _guided_filter(guide, flat, radius=5, eps=1e-3)
    assert out.shape == guide.shape
    np.testing.assert_allclose(out, 0.4, atol=1e-6)


def test_guided_filter_follows_guide_edges():
    from src.enhance.traditional import _guided_filter
    guide = np.zeros((40, 40)); guide[:, 20:] = 1.0
    noisy = guide + 0.05 * np.random.default_rng(3).standard_normal(guide.shape)
    out = _guided_filter(guide, noisy, radius=4, eps=1e-3)
    # 경계는 살리고(좌우 차이 유지) 잡음은 줄인다
    assert out[:, 25:].mean() - out[:, :15].mean() > 0.9
    assert out[:, :15].std() < noisy[:, :15].std()
