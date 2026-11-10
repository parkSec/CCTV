"""P/R/F1, AP@0.5 집계."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.classes import COMMON_CLASSES


@dataclass
class ImageRecord:
    image_id: str
    pred_labels: np.ndarray
    pred_scores: np.ndarray
    is_tp: np.ndarray
    gt_labels: np.ndarray


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


def compute_metrics(records: list[ImageRecord], eval_classes: list[int], conf_eval: float) -> dict[str, dict]:
    labels = np.concatenate([r.pred_labels for r in records]) if records else np.zeros(0, np.int64)
    scores = np.concatenate([r.pred_scores for r in records]) if records else np.zeros(0, np.float32)
    is_tp = np.concatenate([r.is_tp for r in records]) if records else np.zeros(0, bool)
    gts = np.concatenate([r.gt_labels for r in records]) if records else np.zeros(0, np.int64)

    keep = filter_to_classes(labels, eval_classes)
    labels, scores, is_tp = labels[keep], scores[keep], is_tp[keep]
    gts = gts[filter_to_classes(gts, eval_classes)]

    out: dict[str, dict] = {}
    tot_tp = tot_fp = tot_fn = tot_gt = 0
    aps = []
    for c in eval_classes:
        m = labels == c
        n_gt = int(np.sum(gts == c))
        above = scores[m] >= conf_eval
        tp = int(np.sum(is_tp[m] & above))
        fp = int(np.sum(~is_tp[m] & above))
        fn = n_gt - tp
        p, r, f1 = _prf(tp, fp, fn)
        ap = average_precision(scores[m], is_tp[m], n_gt)
        if n_gt > 0:
            aps.append(ap)
        out[COMMON_CLASSES[c]] = {"precision": p, "recall": r, "f1": f1, "ap50": ap,
                                  "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt}
        tot_tp, tot_fp, tot_fn, tot_gt = tot_tp + tp, tot_fp + fp, tot_fn + fn, tot_gt + n_gt
    p, r, f1 = _prf(tot_tp, tot_fp, tot_fn)
    out["all"] = {"precision": p, "recall": r, "f1": f1,
                  "ap50": float(np.mean(aps)) if aps else 0.0,
                  "tp": tot_tp, "fp": tot_fp, "fn": tot_fn, "n_gt": tot_gt}
    return out
