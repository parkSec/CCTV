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


def test_new_datasets_in_config():
    cfg = load_config(DEFAULT)
    assert cfg["datasets"]["acdc_night"]["root"] == PROJECT_ROOT / "datasets/ACDC"
    assert cfg["datasets"]["acdc_night"]["splits"] == ["train", "val"]
    assert cfg["datasets"]["darkdriving"]["root"] == PROJECT_ROOT / "datasets/DarkDriving/Det_data"
    assert cfg["datasets"]["darkdriving"]["split"] == "val"


def test_base_config_inherits_and_overrides(tmp_path):
    p = tmp_path / "child.yaml"
    p.write_text(f"base: {DEFAULT.as_posix()}\nnum_samples: 30\nconditions:\n  - type: original\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["num_samples"] == 30 and [c["type"] for c in cfg["conditions"]] == ["original"]
    assert cfg["detection"]["imgsz"] == [640, 960]            # 기본 설정에서 이어받음
    assert cfg["datasets"]["bdd100k"]["images_dir"] == PROJECT_ROOT / "datasets/bdd100k_hf"
    assert "base" not in cfg


def test_base_relative_to_child(tmp_path):
    (tmp_path / "parent.yaml").write_text(DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    p = tmp_path / "child.yaml"
    p.write_text("base: parent.yaml\nseed: 7\n", encoding="utf-8")
    assert load_config(p)["seed"] == 7


def test_ruas_variants_config():
    cfg = load_config(PROJECT_ROOT / "configs" / "ruas_variants.yaml")
    assert [(c["type"], c.get("variant"), c.get("name")) for c in cfg["conditions"]] == [
        ("original", None, None), ("ruas", "lol", "ruas_lol"), ("ruas", "upe", "ruas_upe"), ("ruas", "dark", "ruas_dark")]
    assert cfg["num_samples"] == 30


def test_ruas_uses_upe_weights():
    cfg = load_config(DEFAULT)
    assert next(c for c in cfg["conditions"] if c["type"] == "ruas")["variant"] == "upe"


def test_traditional_config_has_only_traditional_methods():
    cfg = load_config(PROJECT_ROOT / "configs" / "traditional.yaml")
    assert [c["type"] for c in cfg["conditions"]] == [
        "original", "gamma", "clahe", "he", "agcwd", "msrcr", "lime", "dong", "bimef"]
    # 데이터셋·탐지 모델·지표는 default.yaml과 같다 (팀원 모델 결과와 같은 기준으로 비교)
    base = load_config(DEFAULT)
    for key in ["seed", "num_samples", "datasets", "detectors", "detection", "iqa", "preprocess"]:
        assert cfg[key] == base[key], key


def test_traditional_config_builds_without_skips():
    from src.enhance.registry import build_enhancers
    cfg = load_config(PROJECT_ROOT / "configs" / "traditional.yaml")
    enhancers, skipped = build_enhancers(cfg["conditions"])
    assert skipped == []
    assert [e.name for e in enhancers] == [
        "original", "gamma_g1.5", "clahe_c2.0", "he", "agcwd_a0.5", "msrcr", "lime", "dong", "bimef"]


def test_traditional_sweep_config_builds_unique_conditions():
    from src.enhance.registry import build_enhancers
    cfg = load_config(PROJECT_ROOT / "configs" / "traditional_sweep.yaml")
    enhancers, skipped = build_enhancers(cfg["conditions"])
    assert skipped == []
    assert [e.name for e in enhancers] == [
        "original", "gamma_g1.2", "gamma_g1.5", "gamma_g2.0", "gamma_g2.5",
        "clahe_c1.0", "clahe_c2.0", "clahe_c4.0", "agcwd_a0.25", "agcwd_a0.5", "agcwd_a0.75"]
    base = load_config(DEFAULT)
    for key in ["seed", "num_samples", "datasets", "detectors", "detection", "iqa", "preprocess"]:
        assert cfg[key] == base[key], key
