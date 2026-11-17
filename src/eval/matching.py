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


# AP50-95용 IoU 임계값 10개 (0.50 ~ 0.95, 0.05 간격)
IOU_THRS = np.round(np.arange(0.50, 0.96, 0.05), 2)

# 객체 크기 구간 (넓이, COCO 규칙: small < 32², medium < 96², large ≥ 96²)
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2


def match_image_multi(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                      iou_thrs=IOU_THRS) -> np.ndarray:
    """임계값마다 점수 내림차순 그리디 매칭(같은 클래스, 정답 하나당 한 번).
    반환: (N_pred, T) 매칭된 정답 인덱스, 없으면 -1."""
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)
    iou_thrs = np.atleast_1d(np.asarray(iou_thrs, dtype=np.float64))
    out = np.full((len(pred_labels), len(iou_thrs)), -1, dtype=np.int64)
    if len(pred_labels) == 0 or len(gt_labels) == 0:
        return out
    ious = box_iou(pred_boxes, gt_boxes)
    ious[pred_labels[:, None] != gt_labels[None, :]] = -1.0
    order = np.argsort(-np.asarray(pred_scores), kind="stable")
    for t, thr in enumerate(iou_thrs):
        used = np.zeros(len(gt_labels), dtype=bool)
        for i in order:
            cand = np.where(used, -1.0, ious[i])
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                out[i, t] = j
                used[j] = True
    return out


def size_bucket(boxes: np.ndarray) -> np.ndarray:
    """xyxy 박스 넓이 → "small" | "medium" | "large"."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return np.where(area < SMALL_MAX, "small", np.where(area < MEDIUM_MAX, "medium", "large"))


def match_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr: float = 0.5) -> np.ndarray:
    """IoU 하나 기준 예측별 TP 여부 (match_image_multi의 한 열)."""
    return match_image_multi(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, [iou_thr])[:, 0] >= 0
