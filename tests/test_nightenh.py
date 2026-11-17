import importlib.util

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.enhance.base import EnhancerUnavailable

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = PROJECT_ROOT / "third_party" / "nightenh"


def test_weights_path_by_variant(tmp_path):
    from src.enhance.nightenh import weights_path
    assert weights_path(tmp_path, "delighteffects") == \
        tmp_path / "results" / "delighteffects" / "model" / "delighteffects_params_0600000.pt"
    assert weights_path(tmp_path, "lol") == tmp_path / "results" / "LOL" / "model" / "LOL_params_0900000.pt"


def test_bad_variant_raises():
    from src.enhance.nightenh import NightEnhEnhancer
    with pytest.raises(ValueError, match="night"):
        NightEnhEnhancer(REPO, variant="night")


def test_missing_raises_unavailable_with_weights_location(tmp_path):
    from src.enhance.nightenh import NightEnhEnhancer
    (tmp_path / "repo").mkdir()
    with pytest.raises(EnhancerUnavailable, match="delighteffects_params_0600000.pt"):
        NightEnhEnhancer(tmp_path / "repo", variant="delighteffects", device="cpu")


needs_repo = pytest.mark.skipif(
    not (HAS_TORCH and (REPO / "results" / "delighteffects" / "model" / "delighteffects_params_0600000.pt").exists()),
    reason="torch·Night-Enhancement 저장소와 가중치 필요")


@needs_repo
def test_contract():
    from src.enhance.nightenh import NightEnhEnhancer
    from tests.enhancer_contract import assert_enhancer_contract
    assert_enhancer_contract(NightEnhEnhancer(REPO, "delighteffects"))


@needs_repo
def test_loads_expected_key():
    import torch
    from src.enhance.nightenh import weights_path
    params = torch.load(weights_path(REPO, "delighteffects"), map_location="cpu")
    assert "genA2B" in params
