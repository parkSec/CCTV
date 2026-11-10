"""설정의 conditions 목록 → Enhancer 목록."""
from __future__ import annotations

import itertools

from .base import Enhancer, EnhancerUnavailable, IdentityEnhancer
from .clahe import ClaheEnhancer
from .gamma import GammaEnhancer


def _build_zerodce(cond: dict) -> Enhancer:
    from .zerodce import ZeroDCEEnhancer
    return ZeroDCEEnhancer(cond["weights"], scale_factor=cond.get("scale_factor", 1), device=cond.get("device"))


def _build_realesrgan(cond: dict) -> Enhancer:
    from .realesrgan import RealESRGANEnhancer
    return RealESRGANEnhancer(cond["weights"], scale=cond.get("scale", 2), tile=cond.get("tile", 400),
                              min_tile=cond.get("min_tile", 128), half=cond.get("half", True),
                              device=cond.get("device"))


BUILDERS = {
    "original": lambda c: IdentityEnhancer(),
    "gamma": lambda c: GammaEnhancer(c["gamma"]),
    "clahe": lambda c: ClaheEnhancer(c["clip_limit"], c.get("tile_grid", 8)),
    "zerodce": _build_zerodce,
    "realesrgan": _build_realesrgan,
}


def _expand(cond: dict) -> list[dict]:
    """sweep의 값 조합마다 조건 하나씩 만든다."""
    sweep = cond.get("sweep") or {}
    base = {k: v for k, v in cond.items() if k != "sweep"}
    if not sweep:
        return [base]
    keys = list(sweep)
    return [{**base, **dict(zip(keys, values))} for values in itertools.product(*(sweep[k] for k in keys))]


def build_enhancers(conditions: list[dict]) -> tuple[list[Enhancer], list[dict]]:
    enhancers: list[Enhancer] = []
    skipped: list[dict] = []
    for cond in conditions:
        ctype = cond.get("type")
        if ctype not in BUILDERS:
            raise ValueError(f"알 수 없는 조건 type: {ctype!r} (가능: {sorted(BUILDERS)})")
        for c in _expand(cond):
            try:
                enhancers.append(BUILDERS[ctype](c))
            except EnhancerUnavailable as e:
                skipped.append({"condition": ctype, "reason": str(e)})
    return enhancers, skipped
