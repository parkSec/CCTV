"""실험 설정(YAML) 로더."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_KEYS = ["seed", "num_samples", "datasets", "detector", "conditions"]

DEFAULTS = {
    "matching": {"iou": 0.5},
    "timing": {"warmup": 5},
    "output": {"root": "results", "save_samples": 8},
}

# 프로젝트 루트 기준 절대 경로로 바꿀 키 이름
PATH_KEYS = {"labels_json", "images_dir", "annos_dir", "weights", "root"}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_paths(node):
    if isinstance(node, dict):
        return {
            k: (_to_path(v) if k in PATH_KEYS and isinstance(v, str) else _resolve_paths(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_resolve_paths(v) for v in node]
    return node


def _to_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"설정 파일 {path}에 필수 키가 없습니다: {missing}")
    if not raw["conditions"]:
        raise ConfigError("conditions가 비어 있습니다")
    cfg = _merge(DEFAULTS, raw)
    return _resolve_paths(cfg)
