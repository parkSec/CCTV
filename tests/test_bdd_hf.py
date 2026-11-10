import json

import cv2
import numpy as np

from scripts.download_bdd100k_hf import fiftyone_to_bdd
from src.data.bdd100k import BDD100KDataset


def _write_fiftyone(root, samples):
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "samples.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")


def _sample(path, timeofday, dets, width=1000, height=500):
    s = {
        "filepath": path,
        "timeofday": {"_cls": "Classification", "label": timeofday},
        "detections": {"_cls": "Detections", "detections": [
            {"_cls": "Detection", "label": lab, "bounding_box": box} for lab, box in dets]},
    }
    if width:
        s["metadata"] = {"_cls": "ImageMetadata", "width": width, "height": height}
    return s


def test_converts_relative_boxes_to_pixels(tmp_path):
    _write_fiftyone(tmp_path, [_sample("data/a.jpg", "night", [("car", [0.1, 0.2, 0.3, 0.4])])])
    out = fiftyone_to_bdd(tmp_path)
    [e] = json.loads(out.read_text(encoding="utf-8"))
    assert e["name"] == "data/a.jpg"
    assert e["attributes"]["timeofday"] == "night"
    assert e["labels"][0]["category"] == "car"
    box = e["labels"][0]["box2d"]
    assert [box["x1"], box["y1"], box["x2"], box["y2"]] == [100.0, 100.0, 400.0, 300.0]


def test_missing_metadata_reads_image_size(tmp_path):
    _write_fiftyone(tmp_path, [_sample("data/b.jpg", "night", [("person", [0.5, 0.5, 0.5, 0.5])], width=None)])
    cv2.imwrite(str(tmp_path / "data" / "b.jpg"), np.zeros((40, 80, 3), np.uint8))
    [e] = json.loads(fiftyone_to_bdd(tmp_path).read_text(encoding="utf-8"))
    box = e["labels"][0]["box2d"]
    assert [box["x1"], box["y1"], box["x2"], box["y2"]] == [40.0, 20.0, 80.0, 40.0]


def test_absolute_filepath_made_relative(tmp_path):
    """FiftyOne이 내보낼 때 절대 경로가 남아 있어도 data/ 이하 상대 경로로 바꾼다."""
    _write_fiftyone(tmp_path, [_sample("/home/someone/fo/data/c.jpg", "daytime", [])])
    [e] = json.loads(fiftyone_to_bdd(tmp_path).read_text(encoding="utf-8"))
    assert e["name"] == "data/c.jpg"
    assert e["labels"] == []


def test_no_detections_field(tmp_path):
    s = _sample("data/d.jpg", "night", [])
    s["detections"] = None
    _write_fiftyone(tmp_path, [s])
    [e] = json.loads(fiftyone_to_bdd(tmp_path).read_text(encoding="utf-8"))
    assert e["labels"] == []


def test_output_loads_with_bdd_loader(tmp_path):
    _write_fiftyone(tmp_path, [
        _sample("data/n.jpg", "night", [("pedestrian", [0, 0, 0.1, 0.1]), ("traffic light", [0, 0, 0.1, 0.1])]),
        _sample("data/d.jpg", "daytime", [("car", [0, 0, 0.1, 0.1])]),
    ])
    out = fiftyone_to_bdd(tmp_path)
    [s] = BDD100KDataset(out, tmp_path, timeofday="night").load()
    assert s.image_path == tmp_path / "data" / "n.jpg"
    assert s.labels.tolist() == [0]


def test_download_skips_unneeded_files(tmp_path, monkeypatch):
    """변환에 필요한 samples.json과 data/ 이미지만 받는다 (fields/ 등은 제외해 파일 수를 줄임)."""
    import sys
    import types

    calls = {}
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda **kw: calls.update(kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    from scripts.download_bdd100k_hf import download
    download(tmp_path)
    assert calls["repo_id"] == "dgural/bdd100k" and calls["repo_type"] == "dataset"
    assert "samples.json" in calls["allow_patterns"] and "data/*" in calls["allow_patterns"]
    assert not any(p.startswith("fields") for p in calls["allow_patterns"])
    assert calls["max_workers"] >= 16
