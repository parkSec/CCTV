"""결과 저장과 실행 환경 기록."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path


def relativize_paths(obj, root: Path, home: Path | None = None):
    """결과 파일에 사용자 이름이 담긴 절대 경로가 남지 않도록 경로를 바꾼다.
    프로젝트 안 경로 → 프로젝트 기준 상대 경로, 홈 폴더 안 경로 → '~/...', 나머지는 그대로."""
    home = Path.home() if home is None else home
    if isinstance(obj, dict):
        return {k: relativize_paths(v, root, home) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [relativize_paths(v, root, home) for v in obj]
    if isinstance(obj, Path):
        for base, prefix in ((Path(root), ""), (Path(home), "~/")):
            try:
                return prefix + obj.relative_to(base).as_posix()
            except ValueError:
                continue
        return obj.as_posix()
    if isinstance(obj, str):
        # 문장 속 경로: 프로젝트 경로를 먼저 지우고(프로젝트가 홈 안에 있을 수 있으므로), 그다음 홈을 '~'로
        for base, repl in ((str(root), "."), (str(home), "~")):
            obj = obj.replace(base, repl)
        return obj
    return obj


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
