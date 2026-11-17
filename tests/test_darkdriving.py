"""DarkDriving 로더. 고정 파일은 실제 night val의 labelme JSON 2개와 YOLO txt 2개."""
import json
import shutil

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.data.darkdriving import DarkDrivingDataset

FX = PROJECT_ROOT / "tests" / "fixtures" / "darkdriving"
A, B = "1_1_1729239248824122368", "1_1_1729239249723686400"


@pytest.fixture
def root(tmp_path):
    """실제 폴더 구조: Det_data/{night,day}/... — B만 낮 이미지가 있다."""
    r = tmp_path / "Det_data"
    for stem in (A, B):
        for sub in ("night/labels_me/val", "night/labels/val", "night/images/val"):
            (r / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(FX / f"{stem}.json", r / "night/labels_me/val")
        shutil.copy(FX / f"{stem}.txt", r / "night/labels/val")
        (r / "night/images/val" / f"{stem}.jpg").write_bytes(b"x")
    (r / "day/images/val").mkdir(parents=True)
    (r / "day/images/val" / f"{B}.jpg").write_bytes(b"x")
    return r


def test_loads_boxes_from_labelme(root):
    ds = DarkDrivingDataset(root, split="val")
    assert ds.name == "darkdriving" and ds.eval_classes == [2]
    s = {x.image_id: x for x in ds.load()}
    assert set(s) == {A, B}
    np.testing.assert_allclose(s[B].boxes, [[1196.54568, 1143.865344, 1218.57768, 1165.201408]], rtol=1e-6)
    assert s[B].labels.tolist() == [2]
    assert s[A].image_path == root / "night/images/val" / f"{A}.jpg"


def test_labelme_matches_yolo(root):
    """YOLO txt(정규화 cx, cy, w, h)를 픽셀로 바꾼 값과 labelme 박스가 거의 같아야 한다."""
    s = {x.image_id: x for x in DarkDrivingDataset(root).load()}
    for stem in (A, B):
        cls, cx, cy, w, h = map(float, (FX / f"{stem}.txt").read_text().split())
        W, H = 2448, 2048
        yolo = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        np.testing.assert_allclose(s[stem].boxes[0], yolo, atol=1.0)


def test_ref_path_only_when_day_image_exists(root):
    s = {x.image_id: x for x in DarkDrivingDataset(root).load()}
    assert s[B].ref_path == root / "day/images/val" / f"{B}.jpg"
    assert s[A].ref_path is None


def test_unknown_label_skipped(root):
    p = root / "night/labels_me/val" / f"{A}.json"
    d = json.loads(p.read_text())
    d["shapes"].append({**d["shapes"][0], "label": "pedestrian"})
    p.write_text(json.dumps(d))
    s = {x.image_id: x for x in DarkDrivingDataset(root).load()}
    assert s[A].labels.tolist() == [2]


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DarkDrivingDataset(tmp_path / "Det_data", split="val").load()
