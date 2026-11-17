import importlib.util

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.enhance.base import EnhancerUnavailable

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = PROJECT_ROOT / "third_party" / "ruas"


def test_bad_variant_raises():
    from src.enhance.ruas import RUASEnhancer
    with pytest.raises(ValueError, match="night"):
        RUASEnhancer(REPO, variant="night")


def test_missing_raises_unavailable(tmp_path):
    from src.enhance.ruas import RUASEnhancer
    with pytest.raises(EnhancerUnavailable):
        RUASEnhancer(tmp_path / "nope", variant="dark", device="cpu")


def test_variant_output_index():
    """공식 test.py: lol은 u_list[-1], upe·dark는 u_list[-2]."""
    from src.enhance.ruas import select_output
    u = ["u0", "u1", "u2"]
    assert select_output(u, "lol") == "u2"
    assert select_output(u, "dark") == "u1"
    assert select_output(u, "upe") == "u1"


needs_repo = pytest.mark.skipif(not (HAS_TORCH and (REPO / "ckpt" / "dark.pt").exists()),
                                reason="torch·RUAS 저장소 필요 (python -m scripts.fetch_third_party)")


@needs_repo
def test_contract():
    from src.enhance.ruas import RUASEnhancer
    from tests.enhancer_contract import assert_enhancer_contract
    assert_enhancer_contract(RUASEnhancer(REPO, "dark"))


@needs_repo
def test_brightens_dark_image():
    from src.enhance.ruas import RUASEnhancer
    img = np.full((64, 96, 3), 20, np.uint8)
    assert RUASEnhancer(REPO, "dark").enhance(img).mean() > img.mean()
