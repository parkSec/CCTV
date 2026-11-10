import numpy as np

from src.data.exdark import ExDarkDataset, parse_bbgt

HEADER = "% bbGt version=3\n"


def _make(tmp_path, cls_dir, img_name, anno_name, anno_text):
    img_dir = tmp_path / "ExDark" / cls_dir
    ann_dir = tmp_path / "ExDark_Annno" / cls_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / img_name).write_bytes(b"x")
    if anno_name is not None:
        (ann_dir / anno_name).write_text(anno_text, encoding="utf-8")


def _ds(tmp_path):
    return ExDarkDataset(images_dir=tmp_path / "ExDark", annos_dir=tmp_path / "ExDark_Annno")


def test_parse_bbgt():
    text = HEADER + "Car 10 20 30 40 0 0 0 0 0 0 0\n\n"
    assert parse_bbgt(text) == [("Car", 10.0, 20.0, 30.0, 40.0)]


def test_xywh_to_xyxy(tmp_path):
    _make(tmp_path, "Car", "a.jpg", "a.jpg.txt", HEADER + "Car 10 20 30 40 0 0 0 0 0 0 0\n")
    [s] = _ds(tmp_path).load()
    np.testing.assert_allclose(s.boxes[0], [10, 20, 40, 60])
    assert s.labels.tolist() == [2]


def test_extension_case_mismatch(tmp_path):
    _make(tmp_path, "People", "b.JPG", "b.jpg.txt", HEADER + "People 1 1 5 5 0 0 0 0 0 0 0\n")
    [s] = _ds(tmp_path).load()
    assert s.image_path.name == "b.JPG"
    assert s.labels.tolist() == [0]


def test_skips_images_without_eval_classes(tmp_path):
    _make(tmp_path, "Cup", "c.png", "c.png.txt", HEADER + "Cup 1 1 5 5 0 0 0 0 0 0 0\n")
    _make(tmp_path, "Car", "d.png", "d.png.txt",
          HEADER + "Cup 1 1 5 5 0 0 0 0 0 0 0\nCar 2 2 4 4 0 0 0 0 0 0 0\n")
    samples = _ds(tmp_path).load()
    assert [s.image_id for s in samples] == ["Car/d.png"]
    assert samples[0].labels.tolist() == [2]   # Cup 박스는 제외


def test_missing_anno_file_skipped(tmp_path):
    _make(tmp_path, "Bus", "e.jpg", None, "")
    _make(tmp_path, "Bus", "f.jpg", "f.jpg.txt", HEADER + "Bus 0 0 10 10 0 0 0 0 0 0 0\n")
    samples = _ds(tmp_path).load()
    assert [s.image_id for s in samples] == ["Bus/f.jpg"]
