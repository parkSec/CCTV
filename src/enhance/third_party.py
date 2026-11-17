"""외부 개선 모델 공식 저장소(third_party/<이름>)에서 모델 클래스를 불러온다.

저장소 여러 개가 모두 `model.py`처럼 같은 이름의 모듈을 갖기 때문에, 불러오는 동안만 저장소 폴더를
sys.path 맨 앞에 넣고, 그 동안 새로 생긴 sys.modules 항목은 끝나면 지운다. 그래야 다음 저장소의
같은 이름 모듈과 섞이지 않는다.
"""
from __future__ import annotations

import contextlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from .base import EnhancerUnavailable

FETCH_HINT = "python -m scripts.fetch_third_party"


@contextlib.contextmanager
def repo_context(repo_dir: Path) -> Iterator[None]:
    """이 블록 안에서만 repo_dir의 모듈을 import할 수 있다. 모델을 만들 때 저장소 코드가 안에서
    다시 import하는 경우가 있으므로, 래퍼는 생성까지 이 블록 안에서 한다."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise EnhancerUnavailable(f"외부 저장소 폴더 없음: {repo_dir} ({FETCH_HINT})")
    before = set(sys.modules)
    path = str(repo_dir)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path)
        for name in set(sys.modules) - before:
            del sys.modules[name]


def _import_attr(repo_dir: Path, module: str, attr: str) -> Any:
    """repo_context 안에서 부른다."""
    shadowed = sys.modules.pop(module, None)   # 같은 이름의 다른 모듈이 이미 있으면 잠깐 치운다
    try:
        mod = importlib.import_module(module)
        if not hasattr(mod, attr):
            raise EnhancerUnavailable(f"{repo_dir}/{module}.py에 {attr}이(가) 없음")
        return getattr(mod, attr)
    except EnhancerUnavailable:
        raise
    except Exception as e:  # 저장소 코드의 import 오류 등
        raise EnhancerUnavailable(f"{repo_dir}/{module}.py를 불러올 수 없음 — {type(e).__name__}: {e}") from e
    finally:
        if shadowed is not None:
            sys.modules[module] = shadowed


def build_in_repo(repo_dir: Path, module: str, attr: str, *args, **kwargs) -> Any:
    """repo_dir/module.py의 attr(클래스)를 저장소 import가 되는 상태에서 만든다.
    생성 중 오류(가중치 파일 문제 등)도 EnhancerUnavailable로 바꾼다."""
    with repo_context(repo_dir):
        cls = _import_attr(repo_dir, module, attr)
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            raise EnhancerUnavailable(f"{repo_dir}/{module}.{attr} 생성 실패 — {type(e).__name__}: {e}") from e


def load_symbol(repo_dir: Path, module: str, attr: str) -> Any:
    """repo_dir/module.py의 attr. 폴더·모듈·이름이 없거나 import가 실패하면 EnhancerUnavailable."""
    with repo_context(repo_dir):
        return _import_attr(repo_dir, module, attr)


def commit_of(repo_dir: Path) -> str | None:
    """repo_dir의 현재 git commit (run_meta.json 기록용). git 저장소가 아니면 None."""
    if not Path(repo_dir).is_dir():
        return None
    try:
        r = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None
