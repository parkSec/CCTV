from pathlib import Path

import numpy as np

from src.data.base import Sample, select_samples
from src.data.bdd100k import BDD100KDataset
from src.data.classes import COMMON_CLASSES

FIX = Path(__file__).parent / "fixtures" / "bdd_mini.json"


def _load():
    ds = BDD100KDataset(labels_json=FIX, images_dir=Path("/imgs"), timeofday="night")
    return {s.image_id: s for s in ds.load()}


def test_common_classes_order():
    assert COMMON_CLASSES == ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def test_filters_night_only():
    samples = _load()
    assert sorted(samples) == ["n1.jpg", "n2.jpg", "n3.jpg"]
    assert samples["n1.jpg"].image_path == Path("/imgs") / "n1.jpg"


def test_maps_both_label_versions():
    samples = _load()
    assert samples["n1.jpg"].labels.tolist() == [0, 2]   # pedestrian, car
    assert samples["n3.jpg"].labels.tolist() == [0, 3]   # person(2018), motor(2018)


def test_drops_unmapped_categories():
    s = _load()["n1.jpg"]
    assert len(s.boxes) == 2  # traffic light 제외
    np.testing.assert_allclose(s.boxes[0], [10, 20, 30, 60])
    assert s.boxes.dtype == np.float32 and s.labels.dtype == np.int64


def test_missing_labels_key_gives_empty():
    s = _load()["n2.jpg"]
    assert s.boxes.shape == (0, 4)
    assert s.labels.shape == (0,)


def _fake(n):
    return [Sample(f"img{i:03d}", Path(f"{i}.jpg"), np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
            for i in range(n)]


def test_select_samples_deterministic():
    pool = _fake(50)
    a = [s.image_id for s in select_samples(pool, 10, seed=42)]
    b = [s.image_id for s in select_samples(list(reversed(pool)), 10, seed=42)]
    assert a == b and len(a) == 10


def test_select_samples_returns_all_when_n_too_large():
    assert len(select_samples(_fake(5), 300, seed=42)) == 5
