"""결과 저장과 실행 환경 기록."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path


def _default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):   # numpy 스칼라
        return o.item()
    raise TypeError(f"JSON으로 바꿀 수 없는 값: {type(o)}")


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    # utf-8-sig: 엑셀에서 한글이 깨지지 않게
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def append_jsonl(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_default) + "\n")


def collect_env() -> dict:
    import cv2
    import numpy as np
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "opencv": cv2.__version__, "torch": None, "ultralytics": None, "gpu": "cpu"}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return env
