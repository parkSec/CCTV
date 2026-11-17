import cv2
import numpy as np
import pytest

from src.frames import extract_frame


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))   # 프레임마다 밝기가 다름
    w.release()
    return p


def test_frame_index(video):
    f = extract_frame(video, frame_index=3)
    assert f.shape == (48, 64, 3)
    assert abs(float(f.mean()) - 60) < 5


def test_time_seconds(video):
    f = extract_frame(video, time_s=0.5)   # 10fps → 5번 프레임
    assert abs(float(f.mean()) - 100) < 5


def test_out_of_range_raises(video):
    with pytest.raises(ValueError, match="10"):
        extract_frame(video, frame_index=10)


def test_needs_exactly_one_position(video):
    with pytest.raises(ValueError):
        extract_frame(video)
    with pytest.raises(ValueError):
        extract_frame(video, frame_index=1, time_s=0.1)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frame(tmp_path / "none.mp4", frame_index=0)


def test_non_ascii_path(tmp_path, video):
    """Windows의 OpenCV는 한글 경로 영상을 못 열어서 임시 ASCII 경로로 복사해 읽는다."""
    import shutil
    k = tmp_path / "바탕 화면" / "영상.avi"
    k.parent.mkdir()
    shutil.copyfile(video, k)
    assert abs(float(extract_frame(k, frame_index=2).mean()) - 40) < 5


def test_compare_frame_script_parses_args():
    from scripts.compare_frame import main
    import pytest as _p
    with _p.raises(SystemExit):
        main(["--input", "x.mp4", "--frame", "1", "--time", "2", "--out", "o.png"])   # 둘 다 주면 거부
