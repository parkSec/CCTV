import cv2
import numpy as np

from src.data.resize import resize_max_side, scale_boxes
from src.imageio import read_image


def test_resize_long_side():
    img = np.zeros((2048, 2448, 3), np.uint8)
    out, s = resize_max_side(img, 1280)
    assert out.shape == (1071, 1280, 3)
    assert s == 1280 / 2448


def test_resize_portrait_long_side_is_height():
    out, s = resize_max_side(np.zeros((2000, 1000, 3), np.uint8), 1280)
    assert out.shape == (1280, 640, 3) and s == 1280 / 2000


def test_no_upscale():
    img = np.full((720, 1280, 3), 7, np.uint8)
    out, s = resize_max_side(img, 1280)
    assert s == 1.0
    assert out.shape == img.shape and np.array_equal(out, img)


def test_scale_boxes():
    out = scale_boxes(np.array([[10, 20, 110, 220]], np.float32), 0.5)
    assert out.dtype == np.float32
    assert np.allclose(out, [[5, 10, 55, 110]])


def test_scale_boxes_empty():
    out = scale_boxes(np.zeros((0, 4), np.float32), 0.5)
    assert out.shape == (0, 4)


def test_read_gray_png_returns_bgr(tmp_path):
    p = tmp_path / "gray.png"
    cv2.imwrite(str(p), np.full((30, 40), 100, np.uint8))
    img = read_image(p)
    assert img.shape == (30, 40, 3) and img.dtype == np.uint8


def test_read_bgra_png_returns_bgr(tmp_path):
    p = tmp_path / "bgra.png"
    cv2.imwrite(str(p), np.full((30, 40, 4), 100, np.uint8))
    img = read_image(p)
    assert img.shape == (30, 40, 3) and img.dtype == np.uint8


def test_read_16bit_png_returns_uint8(tmp_path):
    p = tmp_path / "deep.png"
    cv2.imwrite(str(p), np.full((30, 40, 3), 40000, np.uint16))
    img = read_image(p)
    assert img.shape == (30, 40, 3) and img.dtype == np.uint8
