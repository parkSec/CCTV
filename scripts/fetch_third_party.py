"""외부 개선 모델 공식 저장소를 third_party/에 받고, third_party_manifest.json의 commit으로 고정한다.

사용: python -m scripts.fetch_third_party [--only zerodce sci ...]
third_party/는 git에서 제외되어 있어 우리 저장소에 올라가지 않는다.
Night-Enhancement 가중치는 Dropbox에 있어 자동으로 받지 않는다. 놓을 위치를 출력한다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.enhance.third_party import commit_of

MANIFEST = PROJECT_ROOT / "third_party_manifest.json"

NOTES = {
    "nightenh": ("가중치는 README의 Dropbox 링크에서 직접 받아 다음 위치에 둔다:\n"
                 "  third_party/nightenh/results/delighteffects/model/delighteffects_params_0600000.pt\n"
                 "  third_party/nightenh/results/LOL/model/LOL_params_0900000.pt"),
}


def _git(*args, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def fetch_repo(url: str, commit: str, dst: Path) -> str:
    """"cloned" | "updated" | "exists". 고정 commit으로 checkout할 수 없으면 RuntimeError."""
    dst = Path(dst)
    if dst.exists():
        if commit_of(dst) == commit:
            return "exists"
        _git("fetch", "-q", "origin", cwd=dst)
        state = "updated"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = _git("clone", "-q", url, str(dst))
        if r.returncode != 0:
            raise RuntimeError(f"clone 실패: {url}\n{r.stderr.strip()}")
        state = "cloned"
    r = _git("-c", "advice.detachedHead=false", "checkout", "-q", commit, cwd=dst)
    if r.returncode != 0:
        raise RuntimeError(f"{url}: commit {commit}로 checkout 실패. third_party_manifest.json의 commit을 "
                           f"`git ls-remote {url} HEAD` 값으로 고친다.\n{r.stderr.strip()}")
    return state


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", help="받을 저장소 이름")
    args = ap.parse_args(argv)
    root = Path(load_config(PROJECT_ROOT / "configs" / "default.yaml")["third_party"]["root"])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, m in manifest.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}: {fetch_repo(m['url'], m['commit'], root / name)} ({m['commit'][:7]})")
        if name in NOTES:
            print(NOTES[name])


if __name__ == "__main__":
    main()
