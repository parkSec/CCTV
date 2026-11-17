import importlib.util

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.enhance.base import EnhancerUnavailable

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = PROJECT_ROOT / "third_party" / "zerodce"
WEIGHTS = REPO / "Zero-DCE_code" / "snapshots" / "Epoch99.pth"


def test_missing_raises_unavailable(tmp_path):
    from src.enhance.zerodce_orig import ZeroDCEOrigEnhancer
    with pytest.raises(EnhancerUnavailable):
        ZeroDCEOrigEnhancer(tmp_path / "nope", tmp_path / "none.pth", device="cpu")


needs_repo = pytest.mark.skipif(not (HAS_TORCH and WEIGHTS.exists()),
                                reason="torch·Zero-DCE 저장소 필요 (python -m scripts.fetch_third_party)")


@needs_repo
def test_contract():
    from src.enhance.zerodce_orig import ZeroDCEOrigEnhancer
    from tests.enhancer_contract import assert_enhancer_contract
    enh = ZeroDCEOrigEnhancer(REPO, WEIGHTS)
    assert enh.name == "zerodce"
    assert_enhancer_contract(enh)


@needs_repo
def test_brightens_dark_image():
    from src.enhance.zerodce_orig import ZeroDCEOrigEnhancer
    img = np.full((64, 96, 3), 20, np.uint8)
    assert ZeroDCEOrigEnhancer(REPO, WEIGHTS).enhance(img).mean() > img.mean()
