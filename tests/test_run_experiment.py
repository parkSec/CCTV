import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data.base import DetectionDataset, make_sample
from src.detect.base import Detections, Detector
from scripts.run_experiment import run_experiment

SIZES = [(40, 60), (50, 70), (60, 80)]  # 이미지마다 크기를 달리해 OracleDetector가 구분


def _gt_for(h, w):
    return [[5, 5, w // 2, h // 2]], [2]


class OracleDetector(Detector):
    """이미지 크기로 정답을 찾아 그대로 돌려준다."""

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        boxes, labels = _gt_for(h, w)
        return Detections(np.array(boxes, np.float32), np.array(labels, np.int64), np.array([0.9], np.float32))


class FakeDataset(DetectionDataset):
    name = "fake"
    eval_classes = [0, 1, 2, 3, 4, 5]

    def __init__(self, samples):
        self._samples = samples

    def load(self):
        return self._samples


def _make_dataset(tmp_path, broken=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samples = []
    for i, (h, w) in enumerate(SIZES):
        p = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(p), rng.integers(0, 60, (h, w, 3), dtype=np.uint8))
        boxes, labels = _gt_for(h, w)
        samples.append(make_sample(f"img{i}", p, boxes, labels))
    if broken:
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"not an image")
        samples.append(make_sample("broken", bad, [[0, 0, 1, 1]], [0]))
    return FakeDataset(samples)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    c = copy.deepcopy(c)
    c["num_samples"] = 10
    c["timing"]["warmup"] = 1
    c["output"]["save_samples"] = 2
    # 가중치가 PC에 있든 없든 같은 결과가 나오도록, 딥러닝 기법은 없는 경로를 가리키게 한다
    for cond in c["conditions"]:
        if "weights" in cond:
            cond["weights"] = tmp_path / "missing.pth"
    return c


def test_smoke_run(tmp_path, cfg):
    out = run_experiment(cfg, detector=OracleDetector(), datasets=[_make_dataset(tmp_path)],
                         out_root=tmp_path / "results", run_id="t1")
    assert out == tmp_path / "results" / "t1"
    for name in ["metrics.csv", "timing.csv", "per_image.jsonl", "run_meta.json"]:
        assert (out / name).exists(), name
    m = pd.read_csv(out / "metrics.csv")
    row = m[(m["condition"] == "original") & (m["class"] == "all")].iloc[0]
    assert row["f1"] == 1.0 and row["dataset"] == "fake"
    assert set(m["condition"]) == {"original", "gamma_g1.5", "gamma_g2.0", "gamma_g2.5",
                                   "clahe_c2.0", "clahe_c3.0", "clahe_c4.0"}
    t = pd.read_csv(out / "timing.csv")
    assert (t["n_images"] == 3).all()
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert [s["condition"] for s in meta["skipped_conditions"]] == ["zerodce", "realesrgan"]
    assert meta["sample_ids"]["fake"] == ["img0", "img1", "img2"]
    assert len(list((out / "samples" / "fake" / "original").glob("*.jpg"))) == 2
    lines = (out / "per_image.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 * 7


def test_unreadable_image_excluded(tmp_path, cfg):
    out = run_experiment(cfg, detector=OracleDetector(), datasets=[_make_dataset(tmp_path, broken=True)],
                         out_root=tmp_path / "results", run_id="t2")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert len(meta["excluded_images"]) == 1
    assert meta["excluded_images"][0]["image_id"] == "broken"
    t = pd.read_csv(out / "timing.csv")
    assert (t["n_images"] == 3).all()


def test_missing_dataset_path_raises(tmp_path, cfg):
    cfg["datasets"]["bdd100k"]["labels_json"] = tmp_path / "nope.json"
    cfg["datasets"]["exdark"]["enabled"] = False
    with pytest.raises(FileNotFoundError, match="nope.json"):
        run_experiment(cfg, detector=OracleDetector(), out_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_unicode_path_images_still_read(tmp_path, cfg, monkeypatch):
    """Windows에서 cv2.imread/imwrite는 한글 경로를 처리하지 못한다. 그 상황을 흉내 낸다."""
    ds = _make_dataset(tmp_path / "바탕 화면")
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    out = run_experiment(cfg, detector=OracleDetector(), datasets=[ds],
                         out_root=tmp_path / "바탕 화면" / "results", run_id="u1")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["excluded_images"] == []
    assert len(list((out / "samples" / "fake" / "original").glob("*.jpg"))) == 2


class BrokenEnhancer:
    name = "broken"
    params = {}
    uses_cuda = False

    def enhance(self, img):
        raise RuntimeError("CUDA error: device-side assert")


def test_condition_failing_at_warmup_is_skipped(tmp_path, cfg, monkeypatch):
    import scripts.run_experiment as rx
    from src.enhance.base import IdentityEnhancer
    monkeypatch.setattr(rx, "build_enhancers", lambda conds: ([BrokenEnhancer(), IdentityEnhancer()], []))
    out = run_experiment(cfg, detector=OracleDetector(), datasets=[_make_dataset(tmp_path)],
                         out_root=tmp_path / "results", run_id="w1")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    [s] = meta["skipped_conditions"]
    assert s["condition"] == "broken" and s["dataset"] == "fake" and "CUDA error" in s["reason"]
    m = pd.read_csv(out / "metrics.csv")
    assert set(m["condition"]) == {"original"}


def test_run_meta_has_no_absolute_project_paths(tmp_path, cfg):
    """run_meta.json에 사용자 이름이 들어간 절대 경로가 남지 않아야 한다."""
    out = run_experiment(cfg, detector=OracleDetector(), datasets=[_make_dataset(tmp_path)],
                         out_root=tmp_path / "results", run_id="p1")
    text = (out / "run_meta.json").read_text(encoding="utf-8")
    meta = json.loads(text)
    assert meta["config"]["datasets"]["bdd100k"]["images_dir"] == "datasets/bdd100k_hf"
    assert str(PROJECT_ROOT) not in text
