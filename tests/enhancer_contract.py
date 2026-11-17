"""개선기 공통 계약. 모든 개선 모델 테스트(Task 3, 7~10)가 쓴다."""
from __future__ import annotations

import numpy as np

from src.enhance.base import Enhancer

TIMING_KEYS = ("pre_ms", "infer_ms", "post_ms", "total_ms")


def assert_enhancer_contract(enh: Enhancer) -> None:
    """홀수 크기(721×1281) BGR 입력 → 출력 shape·dtype이 입력과 같고, 단계별 시간 키 4개가 0 이상."""
    img = np.random.default_rng(0).integers(0, 80, (721, 1281, 3), dtype=np.uint8)
    out = enh.enhance(img)
    assert out.shape == img.shape, f"{enh.name}: 출력 {out.shape} != 입력 {img.shape}"
    assert out.dtype == np.uint8
    out2, t = enh.enhance_timed(img)
    assert out2.shape == img.shape and out2.dtype == np.uint8
    assert set(TIMING_KEYS) <= set(t), f"{enh.name}: 시간 키 {sorted(t)}"
    assert all(t[k] >= 0 for k in TIMING_KEYS)
