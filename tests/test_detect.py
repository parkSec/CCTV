import numpy as np
import pytest

from src.detect.base import Detections
from src.detect.yolo import apply_class_map, build_class_map

# dronefreak/bdd100k-yolo26s 모델 카드의 클래스 순서
BDD_MODEL_NAMES = {0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus", 5: "train",
                   6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign"}


def test_build_class_map_bdd():
    assert build_class_map(BDD_MODEL_NAMES) == {0: 0, 1: 0, 2: 2, 3: 5, 4: 4, 6: 3, 7: 1}


def test_build_class_map_unknown_raises():
    with pytest.raises(ValueError, match="scooter"):
        build_class_map({0: "person", 1: "scooter"})


def test_apply_class_map():
    cmap = build_class_map(BDD_MODEL_NAMES)
    boxes = np.arange(12, dtype=np.float32).reshape(3, 4)
    d = apply_class_map(boxes, np.array([5, 2, 8]), np.array([0.9, 0.8, 0.7], np.float32), cmap)
    assert isinstance(d, Detections)
    assert d.labels.tolist() == [2] and d.labels.dtype == np.int64
    np.testing.assert_allclose(d.scores, [0.8])
    np.testing.assert_allclose(d.boxes, boxes[1:2])


def test_apply_class_map_empty():
    d = apply_class_map(np.zeros((0, 4)), np.zeros(0), np.zeros(0), {0: 0})
    assert d.boxes.shape == (0, 4) and d.labels.shape == (0,) and d.scores.shape == (0,)


def test_detections_speed_defaults_empty():
    d = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32))
    assert d.speed == {}


def test_yolo_bdd_detect_returns_speed():
    pytest.importorskip("ultralytics")
    from src.config import PROJECT_ROOT
    from src.detect.yolo import YoloDetector
    w = PROJECT_ROOT / "weights" / "bdd100k-yolo26s.pt"
    if not w.exists():
        pytest.skip("BDD YOLO26s 가중치 없음 (python -m scripts.download_detectors)")
    det = YoloDetector(w, imgsz=640, conf=0.001, device=None, name="yolo26s_bdd")
    out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert out.boxes.shape[1] == 4
    assert {"preprocess", "inference", "postprocess"} <= set(out.speed)
    assert det.name == "yolo26s_bdd" and det.imgsz == 640
