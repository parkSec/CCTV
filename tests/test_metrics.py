import math

import numpy as np
import pytest

from src.eval.matching import match_image_multi
from src.eval.metrics import ImageRecord, compute_metrics, filter_to_classes

ALL = [0, 1, 2, 3, 4, 5]


def box(i, side=10, h=None):
    """i번째 칸에 놓인 박스. 칸끼리 겹치지 않는다."""
    x = i * 1000
    return [x, 0, x + side, h if h is not None else side]


def rec(pred_boxes, pred_labels, scores, gt_boxes, gt_labels, image_id="x"):
    pb = np.array(pred_boxes, np.float32).reshape(-1, 4)
    gb = np.array(gt_boxes, np.float32).reshape(-1, 4)
    pl, gl = np.array(pred_labels, np.int64), np.array(gt_labels, np.int64)
    ps = np.array(scores, np.float32)
    return ImageRecord(image_id, pb, pl, ps, match_image_multi(pb, pl, ps, gb, gl), gb, gl)


def row(rows, cls, size="all"):
    hits = [r for r in rows if r["class"] == cls and r["size"] == size]
    assert len(hits) == 1, (cls, size)
    return hits[0]


def test_filter_to_classes():
    assert filter_to_classes(np.array([0, 5, 3]), [0, 1, 2, 3, 4]).tolist() == [True, False, True]


def test_metrics_perfect():
    rows = compute_metrics([rec([box(0), box(1)], [0, 2], [0.9, 0.8], [box(0), box(1)], [0, 2])], ALL, 0.25)
    for c in ("person", "car", "all"):
        r = row(rows, c)
        assert r["precision"] == r["recall"] == r["f1"] == 1.0
        assert r["ap50"] == pytest.approx(1.0)


def test_row_keys():
    rows = compute_metrics([rec([], [], [], [], [])], ALL, 0.25)
    assert set(rows[0]) == {"class", "size", "precision", "recall", "f1", "ap50", "ap50_95",
                            "tp", "fp", "fn", "n_gt"}
    assert {r["class"] for r in rows} == {"person", "bicycle", "car", "motorcycle", "bus", "truck", "all"}
    assert row(rows, "all")["precision"] == 0.0 and row(rows, "all")["f1"] == 0.0


def test_conf_eval_threshold():
    rows = compute_metrics([rec([box(0)], [0], [0.1], [box(0)], [0])], ALL, 0.25)
    r = row(rows, "person")
    assert r["tp"] == 0 and r["fn"] == 1 and r["recall"] == 0.0
    assert r["ap50"] == pytest.approx(1.0)   # AP는 전체 점수 범위


def test_no_gt_image_all_fp():
    rows = compute_metrics([rec([box(0), box(1)], [2, 2], [0.9, 0.8], [], [])], ALL, 0.25)
    assert row(rows, "car")["fp"] == 2 and row(rows, "car")["n_gt"] == 0
    assert row(rows, "all")["fp"] == 2


def test_class_outside_eval_ignored():
    rows = compute_metrics([rec([box(0), box(1)], [5, 0], [0.9, 0.9], [box(1)], [0])], [0, 1, 2, 3, 4], 0.25)
    assert row(rows, "all")["fp"] == 0 and row(rows, "all")["precision"] == 1.0
    assert "truck" not in {r["class"] for r in rows}


def test_ap_known_value():
    # 점수 순서: TP, FP, TP / 정답 2개
    r = rec([box(0), box(5), box(1)], [0, 0, 0], [0.9, 0.8, 0.7], [box(0), box(1)], [0, 0])
    assert row(compute_metrics([r], ALL, 0.25), "person")["ap50"] == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_map_averages_classes_with_gt_only():
    r = rec([box(0), box(5)], [0, 2], [0.9, 0.9], [box(0), box(1)], [0, 2])   # person AP 1, car AP 0
    assert row(compute_metrics([r], ALL, 0.25), "all")["ap50"] == pytest.approx(0.5)


def test_micro_over_images():
    recs = [rec([box(0)], [0], [0.9], [box(0)], [0], "a"), rec([box(5)], [0], [0.9], [box(0)], [0], "b")]
    r = row(compute_metrics(recs, ALL, 0.25), "all")
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == r["recall"] == r["f1"] == 0.5


def test_ap50_95_perfect():
    r = row(compute_metrics([rec([box(0)], [2], [0.9], [box(0)], [2])], ALL, 0.25), "car")
    assert r["ap50"] == pytest.approx(1.0) and r["ap50_95"] == pytest.approx(1.0)


def test_ap50_95_partial():
    # IoU 0.8: 0.50~0.80의 7개 임계값에서만 맞음 → 7/10
    r = row(compute_metrics([rec([box(0, h=8)], [2], [0.9], [box(0)], [2])], ALL, 0.25), "car")
    assert r["ap50"] == pytest.approx(1.0) and r["ap50_95"] == pytest.approx(0.7)


def test_empty_class_gives_nan_and_is_excluded():
    rows = compute_metrics([rec([box(0)], [2], [0.9], [box(0)], [2])], ALL, 0.25)
    bus = row(rows, "bus")
    assert bus["n_gt"] == 0 and math.isnan(bus["ap50"]) and math.isnan(bus["ap50_95"])
    assert row(rows, "all")["ap50"] == row(rows, "car")["ap50"]


def test_size_rows_exist():
    rows = compute_metrics([rec([], [], [], [], [])], ALL, 0.25)
    assert sorted(r["size"] for r in rows if r["class"] == "all") == ["all", "large", "medium", "small"]


def test_size_bucket_counts_gt_by_bucket():
    # 정답: 작은 차 1(20×20), 큰 차 1(200×200). 예측: 작은 차만 맞춤
    r = rec([box(0, 20)], [2], [0.9], [box(0, 20), box(1, 200)], [2, 2])
    rows = compute_metrics([r], ALL, 0.25)
    small, large = row(rows, "car", "small"), row(rows, "car", "large")
    assert small["n_gt"] == 1 and small["tp"] == 1 and small["recall"] == 1.0
    assert large["n_gt"] == 1 and large["tp"] == 0 and large["fp"] == 0 and large["recall"] == 0.0
    assert math.isnan(row(rows, "car", "medium")["ap50"])


def test_size_bucket_unmatched_pred_counts_by_own_size():
    r = rec([box(3, 200)], [2], [0.9], [box(0, 20)], [2])    # 큰 예측 하나, 매칭 없음
    rows = compute_metrics([r], ALL, 0.25)
    assert row(rows, "car", "large")["fp"] == 1
    assert row(rows, "car", "small")["fp"] == 0 and row(rows, "car", "small")["fn"] == 1


def test_size_bucket_pred_matched_to_other_bucket_excluded():
    # 예측은 중간 크기(40×40)지만 작은 정답(30×30)과 IoU 0.56으로 맞음 → medium 구간에서는 빼고 센다
    r = rec([[0, 0, 40, 40]], [2], [0.9], [[0, 0, 30, 30]], [2])
    rows = compute_metrics([r], ALL, 0.25)
    assert row(rows, "car", "small")["tp"] == 1
    med = row(rows, "car", "medium")
    assert med["tp"] == 0 and med["fp"] == 0 and med["n_gt"] == 0
