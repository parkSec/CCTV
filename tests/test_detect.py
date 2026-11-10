import numpy as np
import pytest

from src.detect.base import Detections
from src.detect.yolo import coco_to_common


def test_coco_to_common_maps_and_drops():
    boxes = np.arange(24, dtype=np.float32).reshape(6, 4)
    cls = np.array([0, 2, 5, 7, 9, 16])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    d = coco_to_common(boxes, cls, scores)
    assert isinstance(d, Detections)
    assert d.labels.tolist() == [0, 2, 4, 5]
    np.testing.assert_allclose(d.scores, [0.9, 0.8, 0.7, 0.6])
    np.testing.assert_allclose(d.boxes, boxes[:4])
    assert d.labels.dtype == np.int64 and d.boxes.dtype == np.float32


def test_empty_detections():
    d = coco_to_common(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    assert d.boxes.shape == (0, 4) and d.labels.shape == (0,) and d.scores.shape == (0,)


def test_yolo_smoke():
    pytest.importorskip("ultralytics")
    from src.config import PROJECT_ROOT
    from src.detect.yolo import YoloDetector
    det = YoloDetector(PROJECT_ROOT / "weights" / "yolo26s.pt", imgsz=640, conf=0.001, device=None)
    out = det.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    assert out.boxes.shape[1] == 4
