"""Ultralytics YOLO26 래퍼 (BDD100K로 학습한 dronefreak/bdd100k-yolo26s·m).
결과를 공통 클래스 체계로 바꾸고, Ultralytics가 잰 단계별 시간(speed)을 같이 돌려준다."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from src.data.classes import BDD_MODEL_IGNORED, BDD_MODEL_TO_COMMON

from .base import Detections, Detector


def build_class_map(model_names: dict[int, str]) -> dict[int, int]:
    """모델 클래스 id → 공통 클래스 id. 매핑표에도 무시 목록에도 없는 이름이 있으면 ValueError."""
    unknown = [n for n in model_names.values() if n not in BDD_MODEL_TO_COMMON and n not in BDD_MODEL_IGNORED]
    if unknown:
        raise ValueError(f"탐지 모델에 모르는 클래스 이름이 있음: {unknown} (src/data/classes.py에 추가 필요)")
    return {int(i): BDD_MODEL_TO_COMMON[n] for i, n in model_names.items() if n in BDD_MODEL_TO_COMMON}


def apply_class_map(boxes, cls, scores, class_map: dict[int, int]) -> Detections:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cls = np.asarray(cls).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = np.isin(cls, list(class_map))
    mapped = np.array([class_map[int(c)] for c in cls[keep]], dtype=np.int64)
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
    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.001, device=0, name: str = ""):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics가 없습니다. `pip install -r requirements.txt`로 설치하세요.") from e
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"탐지 가중치 없음: {weights} (python -m scripts.download_detectors)")
        self.model = YOLO(str(weights))
        self.class_map = build_class_map(self.model.names)
        self.imgsz = int(imgsz)
        self.conf = conf
        self.device = _resolve_device(device)
        self.uses_cuda = self.device != "cpu"
        self.weights_name = weights.name
        self.name = name or weights.stem

    def detect(self, img_bgr: np.ndarray) -> Detections:
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        b = r.boxes
        d = apply_class_map(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(), self.class_map)
        d.speed = {k: float(v) for k, v in (r.speed or {}).items()}
        return d
