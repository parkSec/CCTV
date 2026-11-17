"""배치 실험: 데이터셋 × 화질 개선 조건 → 화질 지표 + 탐지(모델 × 입력 크기) → 지표·시간 기록.

사용: python -m scripts.run_experiment [--config configs/default.yaml] [--num-samples N] [--datasets bdd100k exdark]
      python -m scripts.run_experiment --check   # 준비 상태만 확인

개선된 이미지는 조건마다 이미지당 한 번 만들고, 그 이미지를 화질 지표와 탐지 설정 전부에 같이 쓴다.
어떤 조건에서든 실패한 이미지는 그 데이터셋의 모든 조건 집계에서 뺀다(조건끼리 같은 이미지로 비교).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.data.base import DetectionDataset, Sample, select_samples
from src.data.resize import resize_max_side, scale_boxes
from src.detect.base import Detector
from src.enhance.base import Enhancer
from src.enhance.registry import build_enhancers
from src.enhance.third_party import commit_of
from src.eval.iqa import summarize_iqa
from src.eval.matching import match_image_multi
from src.eval.metrics import ImageRecord, compute_metrics
from src.imageio import read_image, write_image
from src.io_utils import append_jsonl, collect_env, relativize_paths, write_csv, write_json
from src.timing import gpu_allocated_mb, gpu_peak_mb, reset_gpu_peak, summarize_ms, timed

REQUIRED_PATHS = {"bdd100k": ["labels_json", "images_dir"], "exdark": ["images_dir", "annos_dir"],
                  "acdc_night": ["root"], "darkdriving": ["root"]}
ENH_KEYS = ("pre_ms", "infer_ms", "post_ms", "total_ms")


# ---------------------------------------------------------------- 준비

def build_datasets(cfg: dict) -> tuple[list[DetectionDataset], list[dict]]:
    """활성화된 데이터셋 중 경로가 있는 것만 만든다. 없으면 오류 대신 건너뛴다 → (데이터셋, 건너뛴 목록)."""
    from src.data.acdc import ACDCDataset
    from src.data.bdd100k import BDD100KDataset
    from src.data.darkdriving import DarkDrivingDataset
    from src.data.exdark import ExDarkDataset

    out, skipped = [], []
    for n, c in cfg["datasets"].items():
        if not c.get("enabled", True):
            continue
        if n not in REQUIRED_PATHS:
            skipped.append({"dataset": n, "reason": "로더가 아직 없음"})
            continue
        missing = [str(c[k]) for k in REQUIRED_PATHS[n] if not Path(c[k]).exists()]
        if missing:
            skipped.append({"dataset": n, "reason": "경로 없음: " + ", ".join(missing)})
            continue
        if n == "bdd100k":
            out.append(BDD100KDataset(c["labels_json"], c["images_dir"], c.get("timeofday", "night")))
        elif n == "exdark":
            out.append(ExDarkDataset(c["images_dir"], c["annos_dir"]))
        elif n == "acdc_night":
            out.append(ACDCDataset(c["root"], "night", tuple(c.get("splits", ("train", "val")))))
        else:
            out.append(DarkDrivingDataset(c["root"], c.get("split", "val")))
    return out, skipped


def build_detectors(cfg: dict) -> list[Detector]:
    """탐지 모델 × 입력 크기마다 탐지기 하나."""
    from src.detect.yolo import YoloDetector
    det = cfg["detection"]
    return [YoloDetector(d["weights"], imgsz=s, conf=det["conf_min"], device=det.get("device", 0), name=d["name"])
            for d in cfg["detectors"] for s in det["imgsz"]]


def build_iqa(cfg: dict):
    """IQAEvaluator, 쓸 수 없으면 None (경고만)."""
    from src.eval.iqa import IQAEvaluator, IQAUnavailable
    c = cfg["iqa"]
    if not c.get("enabled", True):
        return None
    try:
        return IQAEvaluator(c["metrics_nr"], c["metrics_fr"], device=c.get("device", 0))
    except IQAUnavailable as e:
        print(f"[경고] 화질 지표 없이 진행: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------- 이미지 하나

def _safe_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


def _setting(det: Detector) -> str:
    return f"{det.name}_{det.imgsz}"


def draw_boxes(img: np.ndarray, sample: Sample, det, conf_eval: float) -> np.ndarray:
    vis = img.copy()
    for b in sample.boxes.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)        # 정답: 빨강
    for b, s in zip(det.boxes.astype(int), det.scores):
        if s >= conf_eval:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)    # 예측: 초록
    return vis


def load_prepared(s: Sample, max_side: int) -> tuple[np.ndarray | None, Sample, np.ndarray | None]:
    """(줄인 이미지, 박스도 줄인 Sample, 줄인 정답 이미지 또는 None). 읽을 수 없으면 이미지가 None."""
    img = read_image(s.image_path)
    if img is None:
        return None, s, None
    img, k = resize_max_side(img, max_side)
    ref = None
    if s.ref_path is not None:
        ref = read_image(s.ref_path)
        if ref is not None:
            ref, _ = resize_max_side(ref, max_side)
    return img, replace(s, boxes=scale_boxes(s.boxes, k)), ref


def _is_oom(e: Exception) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enhance_with_retry(enh: Enhancer, img: np.ndarray):
    """GPU 메모리 부족이면 캐시를 비우고 한 번만 다시 시도한다."""
    try:
        return enh.enhance_timed(img)
    except Exception as e:
        if not _is_oom(e):
            raise
        _empty_cache()
        return enh.enhance_timed(img)


class WarmupFailed(Exception):
    """조건의 워밍업이 실패해서 이 조건 전체를 건너뛴다."""


# ---------------------------------------------------------------- 조건 하나

def run_condition(ds: DetectionDataset, samples: list[Sample], enh: Enhancer, detectors: list[Detector],
                  iqa, cfg: dict, out_dir: Path, failures: list) -> dict:
    """{"images": {image_id: 결과}, "gpu_peak_mb": 개선 단계 GPU 메모리 증가 최대치(MB, CUDA 없으면 None)}.
    실패한 이미지는 failures에 넣고 결과에서 뺀다."""
    max_side = cfg["preprocess"]["max_side"]
    conf_eval = cfg["detection"]["conf_eval"]
    n_save = cfg["output"]["save_samples"]

    first, _, _ = load_prepared(samples[0], max_side)
    try:
        for _ in range(cfg["timing"]["warmup"]):
            out = enh.enhance(first)
            for det in detectors:
                det.detect(out)
    except Exception as e:
        raise WarmupFailed(f"워밍업 실패 — {type(e).__name__}: {e}") from e

    images, peak = {}, None
    for i, s0 in enumerate(samples):
        img, s, ref = load_prepared(s0, max_side)
        try:
            # GPU 메모리는 개선 단계에서 늘어난 최대치만 잰다 (탐지·화질 지표 모델이 잡은 메모리는 뺌)
            reset_gpu_peak()
            before = gpu_allocated_mb()
            out, t_enh = enhance_with_retry(enh, img)
            after = gpu_peak_mb()
            if before is not None and after is not None:
                peak = max(peak or 0.0, after - before)
            scores = iqa.score(out, ref) if iqa is not None else {}
            dets = {}
            for det in detectors:
                d, ms = timed(det.detect, out, sync_cuda=det.uses_cuda)
                dets[_setting(det)] = (det, d, ms)
        except Exception as e:  # 한 이미지 실패가 전체 실험을 멈추지 않게
            failures.append({"dataset": ds.name, "image_id": s.image_id, "condition": enh.name,
                             "reason": f"{type(e).__name__}: {e}"})
            continue
        rec = {"enh": t_enh, "iqa": scores, "det": {}}
        for key, (det, d, ms) in dets.items():
            match_idx = match_image_multi(d.boxes, d.labels, d.scores, s.boxes, s.labels)
            above = d.scores >= conf_eval
            rec["det"][key] = {
                "record": ImageRecord(s.image_id, d.boxes, d.labels, d.scores, match_idx, s.boxes, s.labels),
                "ms": ms, "n_pred": int(above.sum()), "n_tp": int(((match_idx[:, 0] >= 0) & above).sum()),
                "n_gt": int(len(s.labels)), "speed": d.speed}
        images[s.image_id] = rec
        if i < n_save and dets:
            _, d0, _ = next(iter(dets.values()))   # 첫 번째 탐지 설정(기준 설정) 결과를 그린다
            p = out_dir / "samples" / ds.name / enh.name / f"{_safe_name(s.image_id)}.jpg"
            write_image(p, draw_boxes(out, s, d0, conf_eval))
    return {"images": images, "gpu_peak_mb": peak}


# ---------------------------------------------------------------- 집계

def aggregate(ds: DetectionDataset, enh: Enhancer, res: dict, keep: list[str], detectors: list[Detector],
              iqa_metrics: list[str], fr_metrics: list[str], has_ref: bool, conf_eval: float):
    imgs = [res["images"][k] for k in keep]
    base = {"dataset": ds.name, "condition": enh.name}
    det_rows = []
    for det in detectors:
        key = _setting(det)
        rows = compute_metrics([r["det"][key]["record"] for r in imgs], ds.eval_classes, conf_eval)
        det_rows += [{**base, "detector": det.name, "imgsz": det.imgsz, "params": json.dumps(enh.params), **r}
                     for r in rows]
    iqa_row = dict(base)
    for m in iqa_metrics:
        if m in fr_metrics and not has_ref:
            iqa_row.update({f"{m}_mean": None, f"{m}_median": None, f"{m}_n_nan": None})
            continue
        st = summarize_iqa([r["iqa"].get(m, float("nan")) for r in imgs])
        iqa_row.update({f"{m}_mean": st["mean"], f"{m}_median": st["median"], f"{m}_n_nan": st["n_nan"]})
    t_row = dict(base)
    for k in ENH_KEYS:
        st = summarize_ms([r["enh"][k] for r in imgs])
        name = k.removesuffix("_ms")
        t_row.update({f"enh_{name}_ms_mean": st["mean"], f"enh_{name}_ms_median": st["median"],
                      f"enh_{name}_ms_p95": st["p95"]})
    for det in detectors:
        st = summarize_ms([r["det"][_setting(det)]["ms"] for r in imgs])
        t_row.update({f"det_{_setting(det)}_ms_mean": st["mean"], f"det_{_setting(det)}_ms_median": st["median"]})
    t_row.update({"gpu_peak_mb": res["gpu_peak_mb"], "n_images": len(imgs)})
    return det_rows, iqa_row, t_row


def _per_image_lines(ds_name, cond, res, excluded_ids):
    for image_id, r in res["images"].items():
        yield {"dataset": ds_name, "condition": cond, "image_id": image_id, "excluded": image_id in excluded_ids,
               "enh": r["enh"], "iqa": r["iqa"],
               "det": {k: {kk: v[kk] for kk in ("ms", "n_pred", "n_tp", "n_gt", "speed")} for k, v in r["det"].items()}}


# ---------------------------------------------------------------- 실험

def run_experiment(cfg: dict, detectors: list[Detector] | None = None, datasets: list[DetectionDataset] | None = None,
                   iqa="auto", out_root: Path | None = None, run_id: str | None = None) -> Path:
    """iqa: IQAEvaluator 같은 객체, None(화질 지표 끔), "auto"(설정으로 만듦)."""
    skipped_datasets = []
    if datasets is None:
        datasets, skipped_datasets = build_datasets(cfg)
    enhancers, skipped = build_enhancers(cfg["conditions"], cfg["third_party"]["root"])
    detectors = detectors if detectors is not None else build_detectors(cfg)
    iqa = build_iqa(cfg) if iqa == "auto" else iqa
    iqa_metrics = (list(iqa.metrics_nr) + list(iqa.metrics_fr)) if iqa is not None else []
    fr_metrics = list(iqa.metrics_fr) if iqa is not None else []
    conf_eval = cfg["detection"]["conf_eval"]

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root or cfg["output"]["root"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")

    det_rows, iqa_rows, timing_rows, excluded, sample_ids = [], [], [], [], {}
    for ds in datasets:
        samples = select_samples(ds.load(), cfg["num_samples"], cfg["seed"])
        readable = []
        for s in samples:
            if read_image(s.image_path) is None:
                excluded.append({"dataset": ds.name, "image_id": s.image_id, "condition": None,
                                 "reason": "이미지를 읽을 수 없음"})
            else:
                readable.append(s)
        sample_ids[ds.name] = [s.image_id for s in readable]
        if not readable:
            print(f"[경고] {ds.name}: 사용할 수 있는 이미지가 없습니다", file=sys.stderr)
            continue
        has_ref = any(s.ref_path is not None for s in readable)
        results, failures = [], []
        for enh in enhancers:
            print(f"[{ds.name}] {enh.name} ({len(readable)}장)")
            try:
                results.append((enh, run_condition(ds, readable, enh, detectors, iqa, cfg, out_dir, failures)))
            except WarmupFailed as e:
                print(f"[건너뜀] {ds.name} / {enh.name}: {e}", file=sys.stderr)
                skipped.append({"condition": enh.name, "dataset": ds.name, "reason": str(e)})
        # 어떤 조건에서든 실패한 이미지는 모든 조건에서 뺀다
        failed_ids = {f["image_id"] for f in failures}
        excluded += failures
        keep = [s.image_id for s in readable if s.image_id not in failed_ids]
        for enh, res in results:
            d, q, t = aggregate(ds, enh, res, keep, detectors, iqa_metrics, fr_metrics, has_ref, conf_eval)
            det_rows += d
            iqa_rows.append(q)
            timing_rows.append(t)
            for line in _per_image_lines(ds.name, enh.name, res, failed_ids):
                append_jsonl(line, out_dir / "per_image.jsonl")

    write_csv(det_rows, out_dir / "detection_metrics.csv")
    write_csv(iqa_rows, out_dir / "iqa_metrics.csv")
    write_csv(timing_rows, out_dir / "timing.csv")
    tp_root = Path(cfg["third_party"]["root"])
    meta = {"run_id": run_id, "started_at": started, "finished_at": datetime.now().isoformat(timespec="seconds"),
            "config": cfg, "env": collect_env(),
            "third_party_commits": {c["repo"]: commit_of(tp_root / c["repo"]) for c in cfg["conditions"] if "repo" in c},
            "detectors": [{"name": d.name, "imgsz": d.imgsz} for d in detectors],
            "iqa": {"metrics": iqa_metrics, "lower_better": getattr(iqa, "lower_better", {})},
            "skipped_datasets": skipped_datasets, "skipped_conditions": skipped,
            "excluded_images": excluded, "sample_ids": sample_ids}
    # 사용자 이름이 담긴 절대 경로를 남기지 않는다
    write_json(relativize_paths(meta, PROJECT_ROOT), out_dir / "run_meta.json")
    for s in skipped_datasets:
        print(f"[건너뜀] 데이터셋 {s['dataset']}: {s['reason']}")
    for s in skipped:
        print(f"[건너뜀] {s['condition']}: {s['reason']}")
    print(f"결과: {out_dir}")
    return out_dir


# ---------------------------------------------------------------- 준비 확인

def _cond_files(cond: dict, root: Path) -> list[Path]:
    repo = root / cond.get("repo", cond["type"])
    t, v = cond["type"], cond.get("variant")
    if t == "zerodce":
        return [Path(cond["weights"])]
    if t == "zerodcepp":
        return [Path(cond["weights"])]
    if t == "sci":
        return [repo / "CVPR" / "weights" / f"{v or 'difficult'}.pt"]
    if t == "ruas":
        return [repo / "ckpt" / f"{v or 'dark'}.pt"]
    if t == "nightenh":
        from src.enhance.nightenh import weights_path
        return [weights_path(repo, v or "delighteffects")]
    return []


def check_ready(cfg: dict) -> list[dict]:
    """실험 전에 데이터셋·가중치·외부 저장소·pyiqa·CUDA가 준비됐는지."""
    rows = []
    ok = lambda b: "OK" if b else "없음"  # noqa: E731
    for n, c in cfg["datasets"].items():
        if not c.get("enabled", True):
            continue
        paths = [Path(c[k]) for k in REQUIRED_PATHS.get(n, [])]
        good = bool(paths) and all(p.exists() for p in paths)
        rows.append({"항목": f"데이터셋 {n}", "상태": ok(good) if paths else "로더 없음",
                     "내용": ", ".join(str(p) for p in paths if not p.exists())})
    for d in cfg["detectors"]:
        rows.append({"항목": f"탐지 {d['name']}", "상태": ok(Path(d["weights"]).exists()), "내용": str(d["weights"])})
    root = Path(cfg["third_party"]["root"])
    for cond in cfg["conditions"]:
        files = _cond_files(cond, root)
        if files:
            miss = [str(f) for f in files if not f.exists()]
            rows.append({"항목": f"개선 {cond['type']}", "상태": ok(not miss), "내용": ", ".join(miss)})
    try:
        import pyiqa
        rows.append({"항목": "pyiqa", "상태": "OK", "내용": getattr(pyiqa, "__version__", "")})
    except ImportError:
        rows.append({"항목": "pyiqa", "상태": "없음", "내용": "pip install pyiqa"})
    try:
        import torch
        cuda = torch.cuda.is_available()
        rows.append({"항목": "CUDA", "상태": ok(cuda), "내용": torch.cuda.get_device_name(0) if cuda else "CPU로 실행됨"})
    except ImportError:
        rows.append({"항목": "CUDA", "상태": "없음", "내용": "torch 없음"})
    return relativize_paths(rows, PROJECT_ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--datasets", nargs="+", help="실행할 데이터셋 이름 (기본: 설정에서 enabled인 것)")
    ap.add_argument("--check", action="store_true", help="준비 상태만 확인하고 끝낸다")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.check:
        for r in check_ready(cfg):
            print(f"{r['상태']:<8} {r['항목']:<24} {r['내용']}")
        return
    if args.num_samples:
        cfg["num_samples"] = args.num_samples
    if args.datasets:
        for name, c in cfg["datasets"].items():
            c["enabled"] = name in args.datasets
    run_experiment(cfg)


if __name__ == "__main__":
    main()
