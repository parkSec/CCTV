import numpy as np
import pytest

from src.eval.matching import box_iou, match_image

B = lambda *rows: np.array(rows, dtype=np.float32).reshape(-1, 4)  # noqa: E731
L = lambda *v: np.array(v, dtype=np.int64)  # noqa: E731
S = lambda *v: np.array(v, dtype=np.float32)  # noqa: E731


def test_iou_identical_is_one():
    assert box_iou(B([0, 0, 10, 10]), B([0, 0, 10, 10]))[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert box_iou(B([0, 0, 10, 10]), B([20, 20, 30, 30]))[0, 0] == 0.0


def test_iou_half_overlap():
    assert box_iou(B([0, 0, 10, 10]), B([5, 0, 15, 10]))[0, 0] == pytest.approx(1 / 3)


def test_iou_shape():
    assert box_iou(B([0, 0, 1, 1], [0, 0, 2, 2]), B()).shape == (2, 0)


def test_match_requires_same_class():
    tp = match_image(B([0, 0, 10, 10]), L(2), S(0.9), B([0, 0, 10, 10]), L(0))
    assert tp.tolist() == [False]


def test_one_gt_matched_once():
    tp = match_image(B([0, 0, 10, 10], [0, 0, 10, 10]), L(2, 2), S(0.6, 0.9),
                     B([0, 0, 10, 10]), L(2))
    assert tp.tolist() == [False, True]   # 점수 높은 두 번째만 TP


def test_greedy_by_score():
    # 예측 2개가 같은 GT와 겹치지만, 점수 높은 예측이 IoU가 낮아도 먼저 가져간다
    gt = B([0, 0, 10, 10])
    preds = B([0, 0, 10, 10], [0, 0, 10, 7])   # IoU 1.0 / 0.7
    tp = match_image(preds, L(1, 1), S(0.5, 0.9), gt, L(1))
    assert tp.tolist() == [False, True]


def test_iou_threshold():
    tp = match_image(B([5, 0, 15, 10]), L(0), S(0.9), B([0, 0, 10, 10]), L(0), iou_thr=0.5)
    assert tp.tolist() == [False]


def test_no_gt():
    tp = match_image(B([0, 0, 1, 1]), L(0), S(0.9), B(), L())
    assert tp.tolist() == [False]


def test_match_multi_thresholds():
    from src.eval.matching import IOU_THRS, match_image_multi
    assert len(IOU_THRS) == 10 and IOU_THRS[0] == 0.5 and IOU_THRS[-1] == 0.95
    m = match_image_multi(B([0, 0, 10, 8]), L(2), S(0.9), B([0, 0, 10, 10]), L(2))   # IoU 0.8
    assert m.shape == (1, 10)
    assert m[0, :7].tolist() == [0] * 7       # 0.50 ~ 0.80
    assert m[0, 7:].tolist() == [-1] * 3      # 0.85 ~ 0.95


def test_match_multi_returns_gt_index():
    from src.eval.matching import match_image_multi
    m = match_image_multi(B([20, 20, 30, 30]), L(0), S(0.9), B([0, 0, 10, 10], [20, 20, 30, 30]), L(0, 0))
    assert (m[:, 0] == 1).all()


def test_match_multi_empty():
    from src.eval.matching import match_image_multi
    assert match_image_multi(B(), L(), S(), B([0, 0, 1, 1]), L(0)).shape == (0, 10)
    assert (match_image_multi(B([0, 0, 1, 1]), L(0), S(0.5), B(), L()) == -1).all()


def test_size_bucket():
    from src.eval.matching import size_bucket
    sides = [31, 32, 95, 97]
    boxes = np.array([[0, 0, s, s] for s in sides], np.float32)
    assert size_bucket(boxes).tolist() == ["small", "medium", "medium", "large"]
