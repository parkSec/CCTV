"""영상의 한 프레임(또는 이미지 한 장)을 조건 8개로 개선·탐지해 비교 격자 이미지로 저장한다.

사용: python -m scripts.compare_frame --input 영상.mp4 --frame 120 --out results/compare.png
      python -m scripts.compare_frame --input 사진.jpg --detector yolo26m_bdd --imgsz 960 --out results/compare.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.data.resize import resize_max_side
from src.enhance.registry import build_enhancers
from src.frames import extract_frame
from src.imageio import read_image, write_image
from src.visualize import compare_frame

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--input", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--frame", type=int)
    g.add_argument("--time", type=float, help="초")
    ap.add_argument("--detector", default="yolo26s_bdd")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    src = Path(args.input)
    if src.suffix.lower() in VIDEO_EXT:
        frame = 0 if args.frame is None and args.time is None else args.frame   # 위치를 안 주면 첫 프레임
        img = extract_frame(src, frame_index=frame, time_s=args.time)
    else:
        img = read_image(src)
        if img is None:
            raise SystemExit(f"이미지를 읽을 수 없음: {src}")
    img, _ = resize_max_side(img, cfg["preprocess"]["max_side"])

    from src.detect.yolo import YoloDetector
    d = next((d for d in cfg["detectors"] if d["name"] == args.detector), None)
    if d is None:
        raise SystemExit(f"설정에 없는 탐지 모델: {args.detector}")
    det = YoloDetector(d["weights"], imgsz=args.imgsz, conf=cfg["detection"]["conf_min"],
                       device=cfg["detection"].get("device", 0), name=d["name"])
    enhancers, skipped = build_enhancers(cfg["conditions"], cfg["third_party"]["root"])
    grid, rows = compare_frame(img, enhancers, det, cfg["detection"]["conf_eval"])
    write_image(Path(args.out), grid)
    for r in rows:
        print(f"{r['condition']:<14} {r['enh_ms']:8.1f} ms  탐지 {r['n_det']}개")
    for s in skipped:
        print(f"[건너뜀] {s['condition']}: {s['reason']}")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
