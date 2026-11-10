"""이미지 입출력. Windows에서 cv2.imread/imwrite는 한글 등 비ASCII 경로를 처리하지 못하므로
바이트로 읽고 쓴 뒤 OpenCV로 디코딩·인코딩한다."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """BGR uint8 이미지. 읽을 수 없으면 None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, img: np.ndarray, quality: int = 90) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
