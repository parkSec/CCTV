import numpy as np
import pytest

from src.eval.metrics import ImageRecord, compute_metrics, filter_to_classes

ALL = [0, 1, 2, 3, 4, 5]


def rec(pred_labels, scores, is_tp, gt_labels, image_id="x"):
    return ImageRecord(
        image_id=image_id,
        pred_labels=np.array(pred_labels, dtype=np.int64),
        pred_scores=np.array(scores, dtype=np.float32),
        is_tp=np.array(is_tp, dtype=bool),
        gt_labels=np.array(gt_labels, dtype=np.int64),
    )


def test_filter_to_classes():
    assert filter_to_classes(np.array([0, 5, 3]), [0, 1, 2, 3, 4]).tolist() == [True, False, True]


def test_metrics_perfect():
    m = compute_metrics([rec([0, 2], [0.9, 0.8], [True, True], [0, 2])], ALL, conf_eval=0.25)
    for key in ("person", "car", "all"):
        assert m[key]["precision"] == 1.0
        assert m[key]["recall"] == 1.0
        assert m[key]["f1"] == 1.0
        assert m[key]["ap50"] == pytest.approx(1.0)


def test_all_keys_present():
    m = compute_metrics([rec([], [], [], [])], ALL, conf_eval=0.25)
    assert set(m) == {"person", "bicycle", "car", "motorcycle", "bus", "truck", "all"}
    assert m["all"]["precision"] == 0.0 and m["all"]["f1"] == 0.0


def test_conf_eval_threshold():
    m = compute_metrics([rec([0], [0.1], [True], [0])], ALL, conf_eval=0.25)
    assert m["person"]["tp"] == 0 and m["person"]["fn"] == 1
    assert m["person"]["recall"] == 0.0
    assert m["person"]["ap50"] == pytest.approx(1.0)   # AP는 전체 점수 범위


def test_no_gt_image_all_fp():
    m = compute_metrics([rec([2, 2], [0.9, 0.8], [False, False], [])], ALL, conf_eval=0.25)
    assert m["car"]["fp"] == 2 and m["car"]["n_gt"] == 0
    assert m["all"]["fp"] == 2


def test_class_outside_eval_ignored():
    m = compute_metrics([rec([5, 0], [0.9, 0.9], [False, True], [0])], [0, 1, 2, 3, 4], conf_eval=0.25)
    assert m["all"]["fp"] == 0
    assert m["all"]["precision"] == 1.0
    assert "truck" not in m


def test_ap_known_value():
    r = rec([0, 0, 0], [0.9, 0.8, 0.7], [True, False, True], [0, 0])
    m = compute_metrics([r], ALL, conf_eval=0.25)
    assert m["person"]["ap50"] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_map_averages_classes_with_gt_only():
    r = rec([0, 2], [0.9, 0.9], [True, False], [0, 2])   # person AP 1, car AP 0, 나머지 GT 없음
    m = compute_metrics([r], ALL, conf_eval=0.25)
    assert m["all"]["ap50"] == pytest.approx(0.5)


def test_micro_over_images():
    recs = [rec([0], [0.9], [True], [0], "a"), rec([0], [0.9], [False], [0], "b")]
    m = compute_metrics(recs, ALL, conf_eval=0.25)
    assert m["all"]["tp"] == 1 and m["all"]["fp"] == 1 and m["all"]["fn"] == 1
    assert m["all"]["precision"] == 0.5 and m["all"]["recall"] == 0.5 and m["all"]["f1"] == 0.5
