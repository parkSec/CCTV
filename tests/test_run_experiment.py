import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import PROJECT_ROOT, load_config
from src.data.base import DetectionDataset, make_sample
from src.detect.base import Detections, Detector
from src.enhance.base import IdentityEnhancer, PassThroughEnhancer
from scripts.run_experiment import main, run_experiment

SIZES = [(40, 60), (50, 70), (60, 80)]  # 이미지마다 크기를 달리해 OracleDetector가 구분


def _gt_for(h, w):
    return [[5, 5, w // 2, h // 2]], [2]


class OracleDetector(Detector):
    """이미지 크기로 정답을 찾아 그대로 돌려준다."""

    def __init__(self, name="det", imgsz=640):
        self.name, self.imgsz = name, imgsz
        self.seen_shapes = []

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.seen_shapes.append((h, w))
        boxes, labels = _gt_for(h, w)
        return Detections(np.array(boxes, np.float32), np.array(labels, np.int64), np.array([0.9], np.float32))


def _four_detectors():
    return [OracleDetector(n, s) for n in ("a", "b") for s in (640, 960)]


class FakeIQA:
    metrics_nr = ["niqe"]
    metrics_fr = ["psnr"]
    lower_better = {"niqe": True, "psnr": False}

    def score(self, img, ref=None):
        out = {"niqe": float(img.mean())}
        if ref is not None:
            out["psnr"] = 30.0
        return out


class FakeDataset(DetectionDataset):
    eval_classes = [0, 1, 2, 3, 4, 5]

    def __init__(self, samples, name="fake"):
        self._samples = samples
        self.name = name

    def load(self):
        return self._samples


def _make_dataset(tmp_path, broken=False, sizes=SIZES, with_ref=False, name="fake"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samples = []
    for i, (h, w) in enumerate(sizes):
        p = tmp_path / f"{name}_img{i}.jpg"
        cv2.imwrite(str(p), rng.integers(0, 60, (h, w, 3), dtype=np.uint8))
        ref = None
        if with_ref:
            ref = tmp_path / f"{name}_ref{i}.jpg"
            cv2.imwrite(str(ref), rng.integers(100, 200, (h, w, 3), dtype=np.uint8))
        boxes, labels = _gt_for(h, w)
        samples.append(make_sample(f"img{i}", p, boxes, labels, ref_path=ref))
    if broken:
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"not an image")
        samples.append(make_sample("broken", bad, [[0, 0, 1, 1]], [0]))
    return FakeDataset(samples, name)


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config(PROJECT_ROOT / "configs" / "default.yaml"))
    c["num_samples"] = 10
    c["timing"]["warmup"] = 1
    c["output"]["save_samples"] = 2
    c["third_party"]["root"] = tmp_path / "third_party"   # 외부 저장소를 받은 PC에서도 같은 결과
    for cond in c["conditions"]:
        if "weights" in cond:
            cond["weights"] = tmp_path / "missing.pth"
    return c


def _run(cfg, tmp_path, datasets, run_id="t", **kw):
    kw.setdefault("detectors", _four_detectors())
    kw.setdefault("iqa", FakeIQA())
    return run_experiment(cfg, datasets=datasets, out_root=tmp_path / "results", run_id=run_id, **kw)


def _meta(out):
    return json.loads((out / "run_meta.json").read_text(encoding="utf-8"))


def _all_rows(det, **eq):
    m = (det["class"] == "all") & (det["size"] == "all")
    for k, v in eq.items():
        m &= det[k] == v
    return det[m]


def test_smoke_outputs(tmp_path, cfg):
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    assert out == tmp_path / "results" / "t"
    for name in ["detection_metrics.csv", "iqa_metrics.csv", "timing.csv", "per_image.jsonl", "run_meta.json"]:
        assert (out / name).exists(), name
    det = pd.read_csv(out / "detection_metrics.csv")
    orig = _all_rows(det, condition="original")
    assert len(orig) == 4
    assert set(zip(orig["detector"], orig["imgsz"])) == {("a", 640), ("a", 960), ("b", 640), ("b", 960)}
    assert (orig["f1"] == 1.0).all()
    assert set(det["condition"]) == {"original", "gamma_g1.5", "clahe_c2.0"}
    t = pd.read_csv(out / "timing.csv")
    assert (t["n_images"] == 3).all()
    for col in ["enh_pre_ms_mean", "enh_infer_ms_p95", "enh_total_ms_median", "det_a_640_ms_mean", "det_b_960_ms_median"]:
        assert col in t.columns, col
    meta = _meta(out)
    assert [s["condition"] for s in meta["skipped_conditions"]] == ["zerodce", "zerodcepp", "sci", "ruas", "nightenh"]
    assert meta["sample_ids"]["fake"] == ["img0", "img1", "img2"]
    assert len(list((out / "samples" / "fake" / "original").glob("*.jpg"))) == 2
    lines = (out / "per_image.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 * 3


def test_missing_dataset_skipped(tmp_path, cfg):
    cfg["datasets"]["bdd100k"]["labels_json"] = tmp_path / "nope.json"
    cfg["datasets"]["exdark"]["images_dir"] = tmp_path / "nope_dir"
    out = run_experiment(cfg, detectors=_four_detectors(), iqa=FakeIQA(), out_root=tmp_path / "results", run_id="m")
    skipped = _meta(out)["skipped_datasets"]
    assert {s["dataset"] for s in skipped} == {"bdd100k", "exdark"}
    assert any("nope.json" in s["reason"] for s in skipped)


def test_unknown_dataset_name_skipped(tmp_path, cfg):
    from scripts.run_experiment import build_datasets
    cfg["datasets"] = {"acdc_night": {"enabled": True, "root": tmp_path}}
    datasets, skipped = build_datasets(cfg)
    assert datasets == [] and skipped[0]["dataset"] == "acdc_night"


class FailOnImage(PassThroughEnhancer):
    """크기가 (50, 70)인 이미지에서만 실패한다."""
    name = "failing"

    def __init__(self, exc=RuntimeError("boom"), times=None):
        self.params, self.exc, self.times = {}, exc, times

    def infer(self, x):
        if x.shape[:2] == (50, 70) and (self.times is None or self.times > 0):
            if self.times is not None:
                self.times -= 1
            raise self.exc
        return x


def _patch_enhancers(monkeypatch, enhancers):
    import scripts.run_experiment as rx
    monkeypatch.setattr(rx, "build_enhancers", lambda conds, root=None: (enhancers, []))


def test_failure_excludes_image_from_all_conditions(tmp_path, cfg, monkeypatch):
    _patch_enhancers(monkeypatch, [IdentityEnhancer(), FailOnImage()])
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    t = pd.read_csv(out / "timing.csv")
    assert (t["n_images"] == 2).all()
    assert set(t["condition"]) == {"original", "failing"}
    ex = _meta(out)["excluded_images"]
    assert [(e["image_id"], e["condition"]) for e in ex] == [("img1", "failing")]
    det = pd.read_csv(out / "detection_metrics.csv")
    assert (_all_rows(det, condition="original")["n_gt"] == 2).all()


class OutOfMemoryError(RuntimeError):
    """torch.cuda.OutOfMemoryError 흉내 (작업 공간에는 torch가 없을 수 있음)."""


def test_oom_retry_once(tmp_path, cfg, monkeypatch):
    cfg["timing"]["warmup"] = 0
    _patch_enhancers(monkeypatch, [IdentityEnhancer(), FailOnImage(OutOfMemoryError("CUDA out of memory"), times=1)])
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    assert _meta(out)["excluded_images"] == []
    assert (pd.read_csv(out / "timing.csv")["n_images"] == 3).all()


def test_resize_applied(tmp_path, cfg):
    dets = _four_detectors()
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path, sizes=[(2000, 3000)])], detectors=dets)
    assert set(dets[0].seen_shapes) == {(853, 1280)}
    # OracleDetector는 받은 이미지 크기로 박스를 만든다 → 정답 박스도 같은 비율로 줄었으면 f1 = 1
    det = pd.read_csv(out / "detection_metrics.csv")
    assert (_all_rows(det)["f1"] == 1.0).all()


def test_fr_only_with_ref(tmp_path, cfg):
    ds = [_make_dataset(tmp_path / "a", with_ref=True, name="withref"), _make_dataset(tmp_path / "b", name="noref")]
    out = _run(cfg, tmp_path, ds)
    iqa = pd.read_csv(out / "iqa_metrics.csv")
    assert iqa[iqa["dataset"] == "withref"]["psnr_mean"].notna().all()
    assert iqa[iqa["dataset"] == "noref"]["psnr_mean"].isna().all()
    assert iqa["niqe_mean"].notna().all() and (iqa["niqe_n_nan"] == 0).all()


def test_iqa_disabled_writes_no_iqa_columns(tmp_path, cfg):
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)], iqa=None)
    meta = _meta(out)
    assert meta["iqa"]["metrics"] == []


def test_no_absolute_user_paths(tmp_path, cfg):
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    text = (out / "run_meta.json").read_text(encoding="utf-8")
    meta = json.loads(text)
    assert meta["config"]["datasets"]["bdd100k"]["images_dir"] == "datasets/bdd100k_hf"
    assert str(PROJECT_ROOT) not in text and str(Path.home()) not in text


def test_unreadable_image_excluded(tmp_path, cfg):
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path, broken=True)])
    ex = _meta(out)["excluded_images"]
    assert [e["image_id"] for e in ex] == ["broken"] and ex[0]["condition"] is None
    assert (pd.read_csv(out / "timing.csv")["n_images"] == 3).all()


def test_unicode_path_images_still_read(tmp_path, cfg, monkeypatch):
    """Windows에서 cv2.imread/imwrite는 한글 경로를 처리하지 못한다. 그 상황을 흉내 낸다."""
    ds = _make_dataset(tmp_path / "바탕 화면")
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    out = run_experiment(cfg, detectors=_four_detectors(), iqa=FakeIQA(), datasets=[ds],
                         out_root=tmp_path / "바탕 화면" / "results", run_id="u1")
    assert _meta(out)["excluded_images"] == []
    assert len(list((out / "samples" / "fake" / "original").glob("*.jpg"))) == 2


class BrokenEnhancer(PassThroughEnhancer):
    name = "broken"
    params = {}

    def infer(self, x):
        raise RuntimeError("CUDA error: device-side assert")


def test_condition_failing_at_warmup_is_skipped(tmp_path, cfg, monkeypatch):
    _patch_enhancers(monkeypatch, [BrokenEnhancer(), IdentityEnhancer()])
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    [s] = _meta(out)["skipped_conditions"]
    assert s["condition"] == "broken" and s["dataset"] == "fake" and "CUDA error" in s["reason"]
    assert set(pd.read_csv(out / "detection_metrics.csv")["condition"]) == {"original"}
    assert _meta(out)["excluded_images"] == []


def test_check_mode_runs_nothing(tmp_path, capsys):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["output"]["root"] = str(tmp_path / "results")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    main(["--config", str(p), "--check"])
    assert not (tmp_path / "results").exists()
    text = capsys.readouterr().out
    for item in ["bdd100k", "yolo26s_bdd", "sci", "pyiqa", "CUDA"]:
        assert item in text, item


def test_gpu_peak_is_enhancement_only(tmp_path, cfg, monkeypatch):
    """탐지·화질 지표 모델이 차지한 메모리는 빼고, 개선 단계에서 늘어난 최대치만 기록한다."""
    import scripts.run_experiment as rx
    monkeypatch.setattr(rx, "gpu_allocated_mb", lambda: 100.0)   # 개선 직전 이미 잡혀 있던 메모리
    monkeypatch.setattr(rx, "gpu_peak_mb", lambda: 130.0)        # 개선 중 최대
    monkeypatch.setattr(rx, "reset_gpu_peak", lambda: None)
    out = _run(cfg, tmp_path, [_make_dataset(tmp_path)])
    assert (pd.read_csv(out / "timing.csv")["gpu_peak_mb"] == 30.0).all()
