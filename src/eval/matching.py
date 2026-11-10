"""예측 박스와 정답 박스 매칭."""
from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy 박스 (N,4), (M,4) → IoU (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr: float = 0.5) -> np.ndarray:
    """점수 내림차순 그리디 매칭. 같은 클래스 + IoU ≥ iou_thr, 정답 하나당 한 번. 반환: 예측별 TP 여부."""
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)
    is_tp = np.zeros(len(pred_labels), dtype=bool)
    if len(pred_labels) == 0 or len(gt_labels) == 0:
        return is_tp
    ious = box_iou(pred_boxes, gt_boxes)
    ious[pred_labels[:, None] != gt_labels[None, :]] = -1.0
    used = np.zeros(len(gt_labels), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores), kind="stable"):
        cand = np.where(used, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            is_tp[i] = True
            used[j] = True
    return is_tp
