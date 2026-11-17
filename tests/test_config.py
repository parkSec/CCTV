from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, ConfigError, load_config

DEFAULT = PROJECT_ROOT / "configs" / "default.yaml"


def test_load_default_config():
    cfg = load_config(DEFAULT)
    assert cfg["seed"] == 42
    assert cfg["num_samples"] == 300
    assert cfg["detection"]["conf_eval"] == 0.25
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


def test_default_config_has_new_sections():
    cfg = load_config(DEFAULT)
    assert cfg["preprocess"]["max_side"] == 1280
    assert [d["name"] for d in cfg["detectors"]] == ["yolo26s_bdd", "yolo26m_bdd"]
    assert cfg["detection"]["imgsz"] == [640, 960]
    assert cfg["iqa"]["metrics_fr"] == ["psnr", "ssim", "lpips"]
    assert cfg["iqa"]["metrics_nr"] == ["niqe", "brisque", "musiq", "clipiqa"]
    assert cfg["timing"]["warmup"] == 10


def test_condition_types():
    cfg = load_config(DEFAULT)
    assert [c["type"] for c in cfg["conditions"]] == [
        "original", "gamma", "clahe", "zerodce", "zerodcepp", "sci", "ruas", "nightenh"]


def test_third_party_root_is_absolute():
    cfg = load_config(DEFAULT)
    assert cfg["third_party"]["root"] == PROJECT_ROOT / "third_party"


def test_missing_detectors_raises(tmp_path):
    text = DEFAULT.read_text(encoding="utf-8")
    start = text.index("detectors:")
    end = text.index("\ndetection:")
    bad = tmp_path / "no_detectors.yaml"
    bad.write_text(text[:start] + text[end + 1:], encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
