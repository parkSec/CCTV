import importlib.util

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.enhance.base import EnhancerUnavailable

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = PROJECT_ROOT / "third_party" / "sci"


def test_bad_variant_raises():
    from src.enhance.sci import SCIEnhancer
    with pytest.raises(ValueError, match="night"):
        SCIEnhancer(REPO, variant="night")


def test_missing_raises_unavailable(tmp_path):
    from src.enhance.sci import SCIEnhancer
    with pytest.raises(EnhancerUnavailable):
        SCIEnhancer(tmp_path / "nope", variant="difficult", device="cpu")


def test_check_all_keys_loaded():
    from src.enhance.sci import missing_model_keys
    assert missing_model_keys({"a": 1, "b": 2}, {"a": 1, "b": 2, "extra": 3}) == []
    assert missing_model_keys({"a": 1, "b": 2}, {"a": 1}) == ["b"]


needs_repo = pytest.mark.skipif(not (HAS_TORCH and (REPO / "CVPR" / "weights" / "difficult.pt").exists()),
                                reason="torch·SCI 저장소 필요 (python -m scripts.fetch_third_party)")


@needs_repo
def test_contract():
    from src.enhance.sci import SCIEnhancer
    from tests.enhancer_contract import assert_enhancer_contract
    assert_enhancer_contract(SCIEnhancer(REPO, "difficult"))


@needs_repo
def test_brightens_dark_image():
    from src.enhance.sci import SCIEnhancer
    img = np.full((64, 96, 3), 20, np.uint8)
    assert SCIEnhancer(REPO, "difficult").enhance(img).mean() > img.mean()
