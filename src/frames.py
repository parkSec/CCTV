"""영상에서 한 프레임을 이미지로 꺼낸다 (결과물은 영상이 아니라 이미지)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _open(path: Path):
    """Windows의 OpenCV는 한글 등 비ASCII 경로의 영상을 열지 못하므로 그때는 임시 ASCII 경로로 복사한다."""
    if str(path).isascii():
        return cv2.VideoCapture(str(path)), None
    tmp = Path(tempfile.mkdtemp()) / ("video" + path.suffix)
    shutil.copyfile(path, tmp)
    return cv2.VideoCapture(str(tmp)), tmp


def extract_frame(video: Path, frame_index: int | None = None, time_s: float | None = None) -> np.ndarray:
    """frame_index 또는 time_s(초) 중 하나만 받는다. 범위를 벗어나면 ValueError(영상 길이 포함)."""
    if (frame_index is None) == (time_s is None):
        raise ValueError("frame_index와 time_s 중 하나만 지정하세요")
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"영상 없음: {video}")
    cap, tmp = _open(video)
    try:
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없음: {video}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        idx = frame_index if frame_index is not None else int(round(time_s * fps))
        if not 0 <= idx < n:
            raise ValueError(f"프레임 {idx}는 범위 밖 (영상 길이 {n}프레임, {n / fps if fps else 0:.1f}초)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"프레임 {idx}를 읽을 수 없음")
        return frame
    finally:
        cap.release()
        if tmp is not None:
            shutil.rmtree(tmp.parent, ignore_errors=True)
