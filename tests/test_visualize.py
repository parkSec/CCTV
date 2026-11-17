import numpy as np

from src.detect.base import Detections
from src.visualize import compare_frame, draw_detections, make_grid
from src.enhance.base import IdentityEnhancer
from src.enhance.gamma import GammaEnhancer


def _det():
    return Detections(np.array([[10, 10, 40, 30], [0, 0, 5, 5]], np.float32), np.array([2, 0], np.int64),
                      np.array([0.9, 0.1], np.float32))


def test_draw_detections_only_above_conf():
    img = np.zeros((60, 80, 3), np.uint8)
    out = draw_detections(img, _det(), conf_eval=0.25)
    assert out.shape == img.shape and img.sum() == 0          # 원본은 그대로
    assert out[10:31, 10:41].sum() > 0                          # 점수 0.9 박스는 그림
    assert out[0:6, 0:6].sum() == 0                             # 점수 0.1 박스는 안 그림


def test_draw_gt_in_red():
    img = np.zeros((60, 80, 3), np.uint8)
    empty = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32))
    out = draw_detections(img, empty, 0.25, gt_boxes=np.array([[20, 20, 50, 50]], np.float32))
    assert out[20, 30, 2] == 255 and out[20, 30, 1] == 0


def test_make_grid_rows():
    tiles = [(f"t{i}", np.zeros((30, 40, 3), np.uint8)) for i in range(5)]
    g = make_grid(tiles, cols=4)
    h_tile = g.shape[0] // 2
    assert g.shape[1] == 4 * 40 and g.shape[0] == 2 * h_tile and h_tile > 30   # 2줄, 타일 위에 글자 띠


class FakeDetector:
    name, imgsz, uses_cuda = "fake", 640, False

    def detect(self, img):
        return _det()


def test_compare_frame_one_tile_per_condition():
    img = np.full((60, 80, 3), 30, np.uint8)
    grid, rows = compare_frame(img, [IdentityEnhancer(), GammaEnhancer(1.5)], FakeDetector(), conf_eval=0.25, cols=4)
    assert [r["condition"] for r in rows] == ["original", "gamma_g1.5"]
    assert all(r["enh_ms"] >= 0 and r["n_det"] == 1 for r in rows)
    assert grid.shape[1] == 4 * 80


def test_compare_frame_excludes_first_call_cost():
    import time
    from src.enhance.base import PassThroughEnhancer

    class SlowFirst(PassThroughEnhancer):
        name, params, calls = "slow", {}, 0

        def infer(self, x):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.2)
            return x

    _, rows = compare_frame(np.zeros((20, 20, 3), np.uint8), [SlowFirst()], FakeDetector(), 0.25)
    assert rows[0]["enh_ms"] < 100
