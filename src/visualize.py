"""비교 이미지: 조건마다 개선 → 탐지 결과를 그려 격자로 붙인다."""
from __future__ import annotations

import cv2
import numpy as np

from src.data.classes import COMMON_CLASSES
from src.detect.base import Detections

PRED, GT = (0, 255, 0), (0, 0, 255)   # BGR: 예측 초록, 정답 빨강
HEADER = 28


def draw_detections(img: np.ndarray, det: Detections, conf_eval: float, gt_boxes=None) -> np.ndarray:
    vis = img.copy()
    if gt_boxes is not None:
        for b in np.asarray(gt_boxes).astype(int).reshape(-1, 4):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), GT, 2)
    for b, c, s in zip(det.boxes.astype(int), det.labels, det.scores):
        if s < conf_eval:
            continue
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), PRED, 2)
        cv2.putText(vis, f"{COMMON_CLASSES[int(c)]} {s:.2f}", (b[0], max(b[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED, 1, cv2.LINE_AA)
    return vis


def make_grid(tiles: list[tuple[str, np.ndarray]], cols: int = 4) -> np.ndarray:
    """각 타일 위에 제목 띠를 붙여 cols개씩 줄을 만든다. 크기가 다르면 첫 타일 크기로 맞춘다."""
    h, w = tiles[0][1].shape[:2]
    cells = []
    for title, img in tiles:
        img = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))
        band = np.full((HEADER, w, 3), 255, np.uint8)
        cv2.putText(band, title, (6, HEADER - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cells.append(np.vstack([band, img]))
    blank = np.full_like(cells[0], 255)
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        row += [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def compare_frame(img: np.ndarray, enhancers: list, detector, conf_eval: float, cols: int = 4,
                  gt_boxes=None) -> tuple[np.ndarray, list[dict]]:
    """(격자 이미지, 조건별 {condition, enh_ms, n_det})."""
    tiles, rows = [], []
    detector.detect(img)   # 탐지기도 미리 한 번
    for enh in enhancers:
        enh.enhance(img)   # 첫 호출 준비 비용(CUDA 초기화 등)이 표시 시간에 들어가지 않게 한 번 미리 돌린다
        out, t = enh.enhance_timed(img)
        det = detector.detect(out)
        n = int((det.scores >= conf_eval).sum())
        tiles.append((f"{enh.name}  {t['total_ms']:.1f} ms  det {n}", draw_detections(out, det, conf_eval, gt_boxes)))
        rows.append({"condition": enh.name, "enh_ms": t["total_ms"], "n_det": n})
    return make_grid(tiles, cols), rows
