from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, ConfigError, load_config

DEFAULT = PROJECT_ROOT / "configs" / "default.yaml"


def test_load_default_config():
    cfg = load_config(DEFAULT)
    assert cfg["seed"] == 42
    assert cfg["num_samples"] == 300
    assert cfg["detector"]["conf_eval"] == 0.25
    assert cfg["conditions"][0]["type"] == "original"


def test_relative_paths_resolved():
    cfg = load_config(DEFAULT)
    p = cfg["datasets"]["bdd100k"]["images_dir"]
    assert isinstance(p, Path)
    assert p.is_absolute()
    # 기본값은 Hugging Face 사본을 scripts/download_bdd100k_hf.py로 받은 위치
    assert p == PROJECT_ROOT / "datasets/bdd100k_hf"
    assert cfg["datasets"]["bdd100k"]["labels_json"] == PROJECT_ROOT / "datasets/bdd100k_hf/bdd_labels.json"


def test_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
