"""Ultralytics YOLO26 래퍼. 결과를 공통 클래스 체계로 바꿔 돌려준다."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from src.data.classes import COCO_TO_COMMON

from .base import Detections, Detector


def coco_to_common(boxes, coco_cls, scores) -> Detections:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    coco_cls = np.asarray(coco_cls).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = np.isin(coco_cls, list(COCO_TO_COMMON))
    mapped = np.array([COCO_TO_COMMON[int(c)] for c in coco_cls[keep]], dtype=np.int64)
    return Detections(boxes=boxes[keep], labels=mapped, scores=scores[keep])


def _resolve_device(device):
    import torch
    if device is None:
        return 0 if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        warnings.warn("CUDA를 사용할 수 없어 CPU로 실행합니다. 처리 시간 결과가 GPU 기준과 다릅니다.")
        return "cpu"
    return device


class YoloDetector(Detector):
    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.001, device=0):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics가 없습니다. `pip install -r requirements.txt`로 설치하세요.") from e
        weights = Path(weights)
        # 파일이 없으면 이름만 넘겨 ultralytics가 공식 가중치를 자동으로 받게 한다.
        source = str(weights) if weights.exists() else weights.name
        self.model = YOLO(source)
        self.imgsz = imgsz
        self.conf = conf
        self.device = _resolve_device(device)
        self.uses_cuda = self.device != "cpu"
        self.weights_name = weights.name

    def detect(self, img_bgr: np.ndarray) -> Detections:
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        b = r.boxes
        return coco_to_common(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())
