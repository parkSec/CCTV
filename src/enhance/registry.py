"""설정의 conditions 목록 → Enhancer 목록."""
from __future__ import annotations

import itertools
from pathlib import Path

from src.config import PROJECT_ROOT

from .base import Enhancer, EnhancerUnavailable, IdentityEnhancer
from .clahe import ClaheEnhancer
from .gamma import GammaEnhancer


def _build_zerodcepp(cond: dict) -> Enhancer:
    from .zerodce import ZeroDCEPPEnhancer
    return ZeroDCEPPEnhancer(cond["weights"], scale_factor=cond.get("scale_factor", 1), device=cond.get("device"))


def _repo(cond: dict, root: Path) -> Path:
    return Path(root) / cond.get("repo", cond["type"])


def _build_zerodce(cond: dict, root: Path) -> Enhancer:
    from .zerodce_orig import ZeroDCEOrigEnhancer
    return ZeroDCEOrigEnhancer(_repo(cond, root), cond["weights"], device=cond.get("device"))


def _build_sci(cond: dict, root: Path) -> Enhancer:
    from .sci import SCIEnhancer
    return SCIEnhancer(_repo(cond, root), cond.get("variant", "difficult"), device=cond.get("device"))


def _build_ruas(cond: dict, root: Path) -> Enhancer:
    from .ruas import RUASEnhancer
    return RUASEnhancer(_repo(cond, root), cond.get("variant", "dark"), device=cond.get("device"))


def _build_nightenh(cond: dict, root: Path) -> Enhancer:
    from .nightenh import NightEnhEnhancer
    return NightEnhEnhancer(_repo(cond, root), cond.get("variant", "delighteffects"), device=cond.get("device"))


# third_party 저장소를 쓰는 빌더 (cond, third_party_root)
REPO_BUILDERS = {"zerodce": _build_zerodce, "sci": _build_sci, "ruas": _build_ruas, "nightenh": _build_nightenh}

BUILDERS = {
    "original": lambda c: IdentityEnhancer(),
    "gamma": lambda c: GammaEnhancer(c["gamma"]),
    "clahe": lambda c: ClaheEnhancer(c["clip_limit"], c.get("tile_grid", 8)),
    "zerodcepp": _build_zerodcepp,
}


def _expand(cond: dict) -> list[dict]:
    """sweep의 값 조합마다 조건 하나씩 만든다."""
    sweep = cond.get("sweep") or {}
    base = {k: v for k, v in cond.items() if k != "sweep"}
    if not sweep:
        return [base]
    keys = list(sweep)
    return [{**base, **dict(zip(keys, values))} for values in itertools.product(*(sweep[k] for k in keys))]


def build_enhancers(conditions: list[dict], third_party_root: Path | None = None) -> tuple[list[Enhancer], list[dict]]:
    root = Path(third_party_root) if third_party_root is not None else PROJECT_ROOT / "third_party"
    builders = {**BUILDERS, **{k: (lambda c, f=f: f(c, root)) for k, f in REPO_BUILDERS.items()}}
    enhancers: list[Enhancer] = []
    skipped: list[dict] = []
    for cond in conditions:
        ctype = cond.get("type")
        if ctype not in builders:
            # 재설계 중 아직 래퍼가 없는 모델은 실험을 멈추지 않고 건너뛴다
            skipped.append({"condition": ctype, "reason": f"아직 구현되지 않음: {ctype!r}"})
            continue
        for c in _expand(cond):
            try:
                enh = builders[ctype](c)
                if c.get("name"):   # 같은 모델을 설정만 바꿔 여러 조건으로 넣을 때 (예: ruas_lol, ruas_dark)
                    enh.name = c["name"]
                enhancers.append(enh)
            except EnhancerUnavailable as e:
                skipped.append({"condition": ctype, "reason": str(e)})
    return enhancers, skipped
