"""외부 개선 모델 래퍼(Task 7~10) 중 torch·저장소 없이 확인할 수 있는 부분."""
import pytest

from src.enhance.base import EnhancerUnavailable
from src.enhance.registry import build_enhancers


@pytest.mark.parametrize("ctype", ["zerodce", "sci", "ruas", "nightenh"])
def test_missing_repo_skipped(tmp_path, ctype):
    cond = {"type": ctype, "repo": ctype}
    if ctype == "zerodce":
        cond["weights"] = tmp_path / "none.pth"
    enh, skipped = build_enhancers([cond], third_party_root=tmp_path / "third_party")
    assert enh == [] and skipped[0]["condition"] == ctype and skipped[0]["reason"]
