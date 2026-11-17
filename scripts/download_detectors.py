"""설정의 탐지 모델 가중치를 Hugging Face에서 받는다.

사용: python -m scripts.download_detectors [--config configs/default.yaml]
각 모델 저장소의 best.pt를 설정의 weights 경로로 복사한다. 이미 있으면 건너뛴다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable

from src.config import PROJECT_ROOT, load_config

FILENAME = "best.pt"


def download_detectors(detectors: list[dict], downloader: Callable | None = None) -> dict[str, str]:
    """{모델 이름: "exists" | "downloaded"}."""
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    done = {}
    for d in detectors:
        dst = Path(d["weights"])
        if dst.exists():
            done[d["name"]] = "exists"
            continue
        src = downloader(repo_id=d["hf_repo"], filename=FILENAME)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        done[d["name"]] = "downloaded"
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for name, state in download_detectors(cfg["detectors"]).items():
        print(f"{name}: {state}")


if __name__ == "__main__":
    main()
