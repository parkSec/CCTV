import importlib.util

import numpy as np
import pytest

from src.enhance.registry import build_enhancers
from src.enhance.zerodce import pad_to_multiple

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_pad_to_multiple_shape():
    img = np.zeros((37, 53, 3), np.uint8)
    padded, (h, w) = pad_to_multiple(img, 12)
    assert padded.shape == (48, 60, 3)
    assert (h, w) == (37, 53)
    np.testing.assert_array_equal(padded[:37, :53], img)


def test_pad_to_multiple_noop_when_divisible():
    img = np.ones((24, 36, 3), np.uint8)
    padded, _ = pad_to_multiple(img, 12)
    assert padded.shape == img.shape


def test_missing_weights_skipped(tmp_path):
    enh, skipped = build_enhancers([{"type": "zerodce", "weights": tmp_path / "none.pth", "scale_factor": 1}])
    assert enh == []
    assert "none.pth" in skipped[0]["reason"]


@pytest.mark.skipif(HAS_TORCH, reason="torch가 설치된 환경에서는 해당 없음")
def test_without_torch_skipped(tmp_path):
    w = tmp_path / "Epoch99.pth"
    w.write_bytes(b"x")
    enh, skipped = build_enhancers([{"type": "zerodce", "weights": w}])
    assert enh == [] and "torch" in skipped[0]["reason"]


# ---- 아래는 PyTorch가 있는 PC에서만 실행 ----

def _random_weights(tmp_path, scale_factor=1):
    import torch
    from src.enhance.zerodce import ZeroDCEPPNet
    net = ZeroDCEPPNet(scale_factor)
    p = tmp_path / "rand.pth"
    torch.save(net.state_dict(), p)
    return p


def test_shape_dtype_preserved(tmp_path):
    pytest.importorskip("torch")
    from src.enhance.zerodce import ZeroDCEEnhancer
    enh = ZeroDCEEnhancer(_random_weights(tmp_path), scale_factor=1, device="cpu")
    img = np.random.default_rng(0).integers(0, 60, (37, 53, 3), dtype=np.uint8)
    out = enh.enhance(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert enh.name == "zerodce"


def test_scale_factor_12_keeps_size(tmp_path):
    pytest.importorskip("torch")
    from src.enhance.zerodce import ZeroDCEEnhancer
    enh = ZeroDCEEnhancer(_random_weights(tmp_path, 12), scale_factor=12, device="cpu")
    img = np.zeros((100, 130, 3), np.uint8)
    assert enh.enhance(img).shape == img.shape


def test_official_param_count(tmp_path):
    """논문 기준 Zero-DCE++는 학습 파라미터 약 10K."""
    pytest.importorskip("torch")
    from src.enhance.zerodce import ZeroDCEPPNet
    n = sum(p.numel() for p in ZeroDCEPPNet(1).parameters())
    assert 9_000 < n < 11_000
