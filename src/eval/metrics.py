"""P/R/F1(IoU 0.5), AP50, AP50-95 집계. 클래스별 · 객체 크기별 행을 만든다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.classes import COMMON_CLASSES

from .matching import IOU_THRS, size_bucket

SIZES = ["all", "small", "medium", "large"]


@dataclass
class ImageRecord:
    image_id: str
    pred_boxes: np.ndarray   # (N, 4)
    pred_labels: np.ndarray  # (N,)
    pred_scores: np.ndarray  # (N,)
    match_idx: np.ndarray    # (N, T) 임계값별 매칭된 정답 인덱스, 없으면 -1
    gt_boxes: np.ndarray     # (M, 4)
    gt_labels: np.ndarray    # (M,)


def filter_to_classes(labels: np.ndarray, eval_classes: list[int]) -> np.ndarray:
    return np.isin(np.asarray(labels), eval_classes)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = _ratio(tp, tp + fp)
    r = _ratio(tp, tp + fn)
    f1 = _ratio(2 * p * r, p + r)
    return p, r, f1


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """all-point 보간 AP (VOC 2010+ 방식)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = is_tp[order].astype(np.float64)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _flatten(records: list[ImageRecord], eval_classes: list[int]) -> dict:
    """모든 이미지의 예측·정답을 한 줄로 이어 붙인다. 정답 크기는 매칭된 정답 기준으로 예측마다 붙인다."""
    T = len(IOU_THRS)
    labels, scores, own_size, tp, gt_size_of_match = [], [], [], [], []
    gt_labels, gt_sizes = [], []
    for r in records:
        gl = np.asarray(r.gt_labels, np.int64).reshape(-1)
        gsz = size_bucket(r.gt_boxes) if len(gl) else np.zeros(0, "<U6")
        pl = np.asarray(r.pred_labels, np.int64).reshape(-1)
        mi = np.asarray(r.match_idx, np.int64).reshape(len(pl), T)
        labels.append(pl)
        scores.append(np.asarray(r.pred_scores, np.float32).reshape(-1))
        own_size.append(size_bucket(r.pred_boxes) if len(pl) else np.zeros(0, "<U6"))
        tp.append(mi >= 0)
        gt_size_of_match.append(np.where(mi >= 0, gsz[np.clip(mi, 0, None)] if len(gsz) else "", ""))
        keep = filter_to_classes(gl, eval_classes)
        gt_labels.append(gl[keep])
        gt_sizes.append(gsz[keep])
    cat = lambda xs, empty: np.concatenate(xs) if xs else empty  # noqa: E731
    d = {
        "labels": cat(labels, np.zeros(0, np.int64)),
        "scores": cat(scores, np.zeros(0, np.float32)),
        "own_size": cat(own_size, np.zeros(0, "<U6")),
        "tp": cat(tp, np.zeros((0, T), bool)),
        "match_size": cat(gt_size_of_match, np.zeros((0, T), "<U6")),
        "gt_labels": cat(gt_labels, np.zeros(0, np.int64)),
        "gt_sizes": cat(gt_sizes, np.zeros(0, "<U6")),
    }
    keep = filter_to_classes(d["labels"], eval_classes)
    for k in ("labels", "scores", "own_size", "tp", "match_size"):
        d[k] = d[k][keep]
    return d


def _class_size_stats(d: dict, c: int, size: str, conf_eval: float) -> dict:
    """한 클래스 · 한 크기 구간의 TP/FP/FN과 임계값별 AP."""
    m = d["labels"] == c
    scores, tp, own, msz = d["scores"][m], d["tp"][m], d["own_size"][m], d["match_size"][m]
    gt_in = d["gt_labels"] == c
    if size != "all":
        gt_in &= d["gt_sizes"] == size
    n_gt = int(gt_in.sum())
    aps = []
    for t in range(tp.shape[1]):
        if size == "all":
            incl = np.ones(len(scores), bool)
        else:
            # 매칭된 정답이 이 구간이면 TP, 매칭이 없고 예측 자체가 이 구간이면 FP, 나머지는 뺀다
            incl = np.where(tp[:, t], msz[:, t] == size, own == size)
        aps.append(average_precision(scores[incl], tp[incl, t], n_gt) if n_gt > 0 else float("nan"))
        if t == 0:
            above = incl & (scores >= conf_eval)
            n_tp = int((tp[:, 0] & above).sum())
            n_fp = int((~tp[:, 0] & above).sum())
    return {"tp": n_tp, "fp": n_fp, "fn": n_gt - n_tp, "n_gt": n_gt,
            "ap50": aps[0], "ap50_95": float(np.mean(aps)) if n_gt > 0 else float("nan")}


def compute_metrics(records: list[ImageRecord], eval_classes: list[int], conf_eval: float) -> list[dict]:
    """행마다 class(이름 또는 "all"), size(all/small/medium/large), precision, recall, f1, ap50, ap50_95,
    tp, fp, fn, n_gt. 정답이 0개인 행의 AP는 NaN이고 "all" 평균에서 뺀다."""
    d = _flatten(records, eval_classes)
    rows = []
    for size in SIZES:
        per = []
        for c in eval_classes:
            st = _class_size_stats(d, c, size, conf_eval)
            p, r, f1 = _prf(st["tp"], st["fp"], st["fn"])
            row = {"class": COMMON_CLASSES[c], "size": size, "precision": p, "recall": r, "f1": f1,
                   "ap50": st["ap50"], "ap50_95": st["ap50_95"],
                   "tp": st["tp"], "fp": st["fp"], "fn": st["fn"], "n_gt": st["n_gt"]}
            rows.append(row)
            per.append(row)
        tp, fp, fn = (sum(r[k] for r in per) for k in ("tp", "fp", "fn"))
        p, r, f1 = _prf(tp, fp, fn)
        with_gt = [r for r in per if r["n_gt"] > 0]
        rows.append({"class": "all", "size": size, "precision": p, "recall": r, "f1": f1,
                     "ap50": float(np.mean([r["ap50"] for r in with_gt])) if with_gt else float("nan"),
                     "ap50_95": float(np.mean([r["ap50_95"] for r in with_gt])) if with_gt else float("nan"),
                     "tp": tp, "fp": fp, "fn": fn, "n_gt": sum(r["n_gt"] for r in per)})
    return rows
