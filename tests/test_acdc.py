"""ACDC 야간 로더. 고정 파일은 실제 night val 라벨에서 6장을 고른 것(세그멘테이션 필드만 뺌)."""
import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.data.acdc import ACDCDataset

FX = PROJECT_ROOT / "tests" / "fixtures" / "acdc"
P, B, C, M, BUS, T = 0, 1, 2, 3, 4, 5


@pytest.fixture
def samples():
    ds = ACDCDataset(FX, condition="night", splits=("val",))
    assert ds.name == "acdc_night" and ds.eval_classes == [0, 1, 2, 3, 4, 5]
    return {s.image_id: s for s in ds.load()}


def _sid(frame):
    return f"night/val/GOPR0351/GOPR0351_frame_{frame}"


def test_all_images_loaded_including_empty(samples):
    assert len(samples) == 6
    empty = samples["night/val/GOPR0356/GOPR0356_frame_000479"]
    assert empty.boxes.shape == (0, 4) and empty.labels.shape == (0,)


def test_bbox_xywh_to_xyxy_and_path(samples):
    s = samples[_sid("000818")]
    np.testing.assert_allclose(s.boxes[0], [843, 561, 859, 573])
    assert s.labels.tolist() == [C, C, C]
    assert s.image_path == FX / "rgb_anon" / "night/val/GOPR0351/GOPR0351_frame_000818_rgb_anon.png"
    assert s.ref_path is None


def test_rider_is_person(samples):
    s = samples[_sid("000055")]
    assert (s.labels == P).sum() == 5 and (s.labels == C).sum() == 2
    s = samples[_sid("000099")]
    assert (s.labels == P).sum() == 6       # person 5 + rider 1


def test_train_dropped(samples):
    s = samples[_sid("000748")]
    assert s.labels.tolist() == [C, C]


def _has_box(s, xyxy):
    return bool(np.any(np.all(np.isclose(s.boxes, xyxy), axis=1)))


def test_crowd_dropped(samples):
    s = samples[_sid("000099")]
    assert not _has_box(s, [580, 572, 713, 593])     # 무리(iscrowd=1) 자전거 박스
    s = samples[_sid("000071")]
    assert not _has_box(s, [1671, 536, 1812, 568])   # 무리 car 박스
    assert sorted(s.labels.tolist()) == [P, B, C]     # rider, bicycle, car만 남음 (crowd car·train 빠짐)


def test_missing_split_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        ACDCDataset(FX, condition="night", splits=("train", "val")).load()
