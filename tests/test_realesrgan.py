import importlib
import importlib.util
import sys
import types

import numpy as np
import pytest

from src.enhance.realesrgan import patch_torchvision_functional_tensor, run_with_tile_retry
from src.enhance.registry import build_enhancers


class FakeOOM(RuntimeError):
    pass


def is_fake_oom(e):
    return isinstance(e, FakeOOM)


def test_tile_retry_halves_until_success():
    tried = []

    def fn(tile):
        tried.append(tile)
        if tile > 100:
            raise FakeOOM()
        return "ok"

    result, tile = run_with_tile_retry(fn, tile=400, min_tile=64, is_oom=is_fake_oom)
    assert result == "ok" and tile == 100
    assert tried == [400, 200, 100]


def test_tile_retry_gives_up_below_min():
    def fn(tile):
        raise FakeOOM()

    with pytest.raises(FakeOOM):
        run_with_tile_retry(fn, tile=400, min_tile=128, is_oom=is_fake_oom)


def test_tile_retry_other_errors_not_retried():
    calls = []

    def fn(tile):
        calls.append(tile)
        raise ValueError("bad")

    with pytest.raises(ValueError):
        run_with_tile_retry(fn, tile=400, min_tile=128, is_oom=is_fake_oom)
    assert calls == [400]


def test_patch_functional_tensor(monkeypatch):
    """torchvision 0.17+ 처럼 functional_tensor가 없을 때 functional을 그 이름으로 등록한다."""
    tv = types.ModuleType("torchvision")
    tr = types.ModuleType("torchvision.transforms")
    fn = types.ModuleType("torchvision.transforms.functional")
    fn.rgb_to_grayscale = lambda x: x
    for name, mod in [("torchvision", tv), ("torchvision.transforms", tr), ("torchvision.transforms.functional", fn)]:
        monkeypatch.setitem(sys.modules, name, mod)
    # 테스트가 끝나면 패치로 생긴 항목까지 원래 상태로 되돌리도록 monkeypatch에 등록해 둔다
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional_tensor", None)
    monkeypatch.delitem(sys.modules, "torchvision.transforms.functional_tensor")

    assert patch_torchvision_functional_tensor() is True
    assert importlib.import_module("torchvision.transforms.functional_tensor").rgb_to_grayscale is fn.rgb_to_grayscale


def test_missing_weights_skipped(tmp_path):
    enh, skipped = build_enhancers([{"type": "realesrgan", "weights": tmp_path / "x2.pth", "scale": 2}])
    assert enh == [] and "x2.pth" in skipped[0]["reason"]


@pytest.mark.skipif(importlib.util.find_spec("realesrgan") is not None, reason="realesrgan 설치 환경")
def test_without_package_skipped(tmp_path):
    w = tmp_path / "RealESRGAN_x2plus.pth"
    w.write_bytes(b"x")
    enh, skipped = build_enhancers([{"type": "realesrgan", "weights": w, "scale": 2}])
    assert enh == [] and "realesrgan" in skipped[0]["reason"]


# ---- PyTorch + realesrgan + 가중치가 있는 PC에서만 실행 ----

def test_output_same_size_as_input():
    pytest.importorskip("realesrgan")
    from src.config import PROJECT_ROOT
    from src.enhance.realesrgan import RealESRGANEnhancer
    w = PROJECT_ROOT / "weights" / "RealESRGAN_x2plus.pth"
    if not w.exists():
        pytest.skip("weights/RealESRGAN_x2plus.pth 없음")
    enh = RealESRGANEnhancer(w, scale=2, tile=200, half=False)
    img = np.random.default_rng(0).integers(0, 60, (90, 130, 3), dtype=np.uint8)
    out = enh.enhance(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert enh.name == "realesrgan_x2"
