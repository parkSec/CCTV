"""배치 실험: 데이터셋 × 화질 개선 조건 → 탐지 → 지표·시간 기록.

사용: python -m scripts.run_experiment --config configs/default.yaml [--num-samples N] [--datasets bdd100k exdark]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.data.base import DetectionDataset, Sample, select_samples
from src.detect.base import Detector
from src.enhance.base import Enhancer
from src.enhance.registry import build_enhancers
from src.eval.matching import match_image_multi
from src.eval.metrics import ImageRecord, compute_metrics
from src.imageio import read_image, write_image
from src.io_utils import append_jsonl, collect_env, relativize_paths, write_csv, write_json
from src.timing import summarize_ms, timed

REQUIRED_PATHS = {"bdd100k": ["labels_json", "images_dir"], "exdark": ["images_dir", "annos_dir"]}


def build_datasets(cfg: dict) -> list[DetectionDataset]:
    """활성화된 데이터셋을 만든다. 경로가 없으면 실험 시작 전에 FileNotFoundError."""
    from src.data.bdd100k import BDD100KDataset
    from src.data.exdark import ExDarkDataset

    dcfg = cfg["datasets"]
    enabled = [n for n, c in dcfg.items() if c.get("enabled", True)]
    missing = [str(dcfg[n][k]) for n in enabled for k in REQUIRED_PATHS.get(n, []) if not Path(dcfg[n][k]).exists()]
    if missing:
        raise FileNotFoundError("데이터셋 경로가 없습니다 (configs/default.yaml 확인): " + ", ".join(missing))
    out = []
    for n in enabled:
        c = dcfg[n]
        if n == "bdd100k":
            out.append(BDD100KDataset(c["labels_json"], c["images_dir"], c.get("timeofday", "night")))
        elif n == "exdark":
            out.append(ExDarkDataset(c["images_dir"], c["annos_dir"]))
        else:
            raise ValueError(f"알 수 없는 데이터셋: {n}")
    return out


def build_detector(cfg: dict) -> Detector:
    from src.detect.yolo import YoloDetector
    # Task 15 전까지는 첫 번째 탐지 모델, 첫 번째 입력 크기만 쓴다
    d, det = cfg["detectors"][0], cfg["detection"]
    return YoloDetector(d["weights"], imgsz=det["imgsz"][0], conf=det["conf_min"], device=det.get("device", 0))


def _safe_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


def draw_boxes(img: np.ndarray, sample: Sample, det, conf_eval: float) -> np.ndarray:
    vis = img.copy()
    for b in sample.boxes.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)        # 정답: 빨강
    for b, s in zip(det.boxes.astype(int), det.scores):
        if s >= conf_eval:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)    # 예측: 초록
    return vis


def _readable(samples: list[Sample], ds_name: str, excluded: list) -> list[Sample]:
    ok = []
    for s in samples:
        if read_image(s.image_path) is None:
            excluded.append({"dataset": ds_name, "image_id": s.image_id, "condition": None,
                             "reason": "이미지를 읽을 수 없음"})
        else:
            ok.append(s)
    return ok


class WarmupFailed(Exception):
    """조건의 워밍업이 실패해서 이 조건 전체를 건너뛴다."""


def run_condition(ds: DetectionDataset, samples: list[Sample], enh: Enhancer, det: Detector,
                  cfg: dict, out_dir: Path, excluded: list) -> tuple[list[dict], dict]:
    conf_eval = cfg["detection"]["conf_eval"]
    n_save = cfg["output"]["save_samples"]

    first = read_image(samples[0].image_path)
    try:
        for _ in range(cfg["timing"]["warmup"]):
            det.detect(enh.enhance(first))
    except Exception as e:
        raise WarmupFailed(f"워밍업 실패 — {type(e).__name__}: {e}") from e

    records, enh_ms, det_ms = [], [], []
    for i, s in enumerate(samples):
        img = read_image(s.image_path)
        try:
            out, t_enh = timed(enh.enhance, img, sync_cuda=enh.uses_cuda)
            d, t_det = timed(det.detect, out, sync_cuda=det.uses_cuda)
        except Exception as e:  # 한 이미지 실패가 전체 실험을 멈추지 않게
            excluded.append({"dataset": ds.name, "image_id": s.image_id, "condition": enh.name,
                             "reason": f"{type(e).__name__}: {e}"})
            continue
        match_idx = match_image_multi(d.boxes, d.labels, d.scores, s.boxes, s.labels)
        is_tp = match_idx[:, 0] >= 0
        records.append(ImageRecord(s.image_id, d.boxes, d.labels, d.scores, match_idx, s.boxes, s.labels))
        enh_ms.append(t_enh)
        det_ms.append(t_det)
        above = d.scores >= conf_eval
        append_jsonl({"dataset": ds.name, "condition": enh.name, "image_id": s.image_id,
                      "n_pred": int(above.sum()), "n_tp": int((is_tp & above).sum()), "n_gt": int(len(s.labels)),
                      "enhance_ms": t_enh, "detect_ms": t_det}, out_dir / "per_image.jsonl")
        if i < n_save:
            p = out_dir / "samples" / ds.name / enh.name / f"{_safe_name(s.image_id)}.jpg"
            write_image(p, draw_boxes(out, s, d, conf_eval))

    metrics = compute_metrics(records, ds.eval_classes, conf_eval)
    rows = [{"dataset": ds.name, "condition": enh.name, "params": json.dumps(enh.params), **vals}
            for vals in metrics]
    e, t = summarize_ms(enh_ms), summarize_ms(det_ms)
    timing = {"dataset": ds.name, "condition": enh.name,
              "enhance_ms_mean": e["mean"], "enhance_ms_median": e["median"],
              "detect_ms_mean": t["mean"], "detect_ms_median": t["median"], "n_images": len(records)}
    return rows, timing


def run_experiment(cfg: dict, detector: Detector | None = None, datasets: list[DetectionDataset] | None = None,
                   out_root: Path | None = None, run_id: str | None = None) -> Path:
    datasets = datasets if datasets is not None else build_datasets(cfg)   # 경로 검사가 가장 먼저
    enhancers, skipped = build_enhancers(cfg["conditions"])
    detector = detector if detector is not None else build_detector(cfg)

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root or cfg["output"]["root"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")

    metric_rows, timing_rows, excluded, sample_ids = [], [], [], {}
    for ds in datasets:
        samples = select_samples(ds.load(), cfg["num_samples"], cfg["seed"])
        samples = _readable(samples, ds.name, excluded)
        sample_ids[ds.name] = [s.image_id for s in samples]
        if not samples:
            print(f"[경고] {ds.name}: 사용할 수 있는 이미지가 없습니다", file=sys.stderr)
            continue
        for enh in enhancers:
            print(f"[{ds.name}] {enh.name} ({len(samples)}장)")
            try:
                rows, timing = run_condition(ds, samples, enh, detector, cfg, out_dir, excluded)
            except WarmupFailed as e:
                print(f"[건너뜀] {ds.name} / {enh.name}: {e}", file=sys.stderr)
                skipped.append({"condition": enh.name, "dataset": ds.name, "reason": str(e)})
                continue
            metric_rows += rows
            timing_rows.append(timing)

    write_csv(metric_rows, out_dir / "metrics.csv")
    write_csv(timing_rows, out_dir / "timing.csv")
    meta = {"run_id": run_id, "started_at": started, "finished_at": datetime.now().isoformat(timespec="seconds"),
            "config": cfg, "env": collect_env(), "skipped_conditions": skipped,
            "excluded_images": excluded, "sample_ids": sample_ids}
    # 사용자 이름이 담긴 절대 경로를 남기지 않는다
    write_json(relativize_paths(meta, PROJECT_ROOT), out_dir / "run_meta.json")
    for s in skipped:
        print(f"[건너뜀] {s['condition']}: {s['reason']}")
    print(f"결과: {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--datasets", nargs="+", help="실행할 데이터셋 이름 (기본: 설정에서 enabled인 것)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.num_samples:
        cfg["num_samples"] = args.num_samples
    if args.datasets:
        for name, c in cfg["datasets"].items():
            c["enabled"] = name in args.datasets
    run_experiment(cfg)


if __name__ == "__main__":
    main()
