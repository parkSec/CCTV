"""BDD100K 검증 세트(Hugging Face 비공식 사본 dgural/bdd100k)를 받아 BDD 라벨 형식으로 변환한다.

사용: python -m scripts.download_bdd100k_hf            (다운로드 + 변환)
      python -m scripts.download_bdd100k_hf --skip-download   (이미 받은 폴더를 변환만)

- 원본은 FiftyOne 형식(samples.json + data/)이다. 박스는 0~1 상대 좌표 [x, y, w, h].
- 변환 결과 `bdd_labels.json`은 src/data/bdd100k.py가 그대로 읽는 형식(name, attributes.timeofday, labels[].box2d).
- 이 데이터는 BDD100K 라이선스(교육·연구·비영리 사용)를 따른다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.imageio import read_image

REPO_ID = "dgural/bdd100k"
DEFAULT_ROOT = PROJECT_ROOT / "datasets" / "bdd100k_hf"


def _relative_path(filepath: str) -> str:
    """FiftyOne filepath를 데이터셋 루트 기준 상대 경로(data/...)로 바꾼다."""
    p = filepath.replace("\\", "/")
    if "/data/" in p:
        return "data/" + p.rsplit("/data/", 1)[1]
    if p.startswith("data/"):
        return p
    return "data/" + p.rsplit("/", 1)[-1]


def _label(field) -> str | None:
    return field.get("label") if isinstance(field, dict) else None


def _image_size(sample: dict, root: Path, rel: str) -> tuple[float, float] | None:
    meta = sample.get("metadata") or {}
    if meta.get("width") and meta.get("height"):
        return float(meta["width"]), float(meta["height"])
    img = read_image(root / rel)
    if img is None:
        return None
    return float(img.shape[1]), float(img.shape[0])


def fiftyone_to_bdd(root: Path, out_name: str = "bdd_labels.json") -> Path:
    root = Path(root)
    with open(root / "samples.json", encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    entries, skipped = [], 0
    for s in samples:
        rel = _relative_path(s["filepath"])
        dets = ((s.get("detections") or {}).get("detections")) or []
        labels = []
        if dets:
            size = _image_size(s, root, rel)
            if size is None:
                skipped += 1
                continue
            w, h = size
            for d in dets:
                bx, by, bw, bh = d["bounding_box"]
                box = {"x1": bx * w, "y1": by * h, "x2": (bx + bw) * w, "y2": (by + bh) * h}
                labels.append({"category": d["label"], "box2d": {k: round(v, 3) for k, v in box.items()}})
        entries.append({"name": rel,
                        "attributes": {"timeofday": _label(s.get("timeofday")),
                                       "weather": _label(s.get("weather")),
                                       "scene": _label(s.get("scene"))},
                        "labels": labels})
    out = root / out_name
    out.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    if skipped:
        print(f"[경고] 이미지 크기를 알 수 없어 제외한 항목: {skipped}개")
    return out


def download(root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("huggingface_hub가 없습니다. `pip install -r requirements.txt`를 먼저 실행하세요.") from e
    print(f"다운로드: {REPO_ID} → {root}")
    # 변환에 필요한 것만 받는다. fields/ 등 다른 폴더는 제외해 파일 수를 줄인다.
    # 중간에 끊겨도 다시 실행하면 이미 받은 파일은 건너뛴다.
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(root),
                      allow_patterns=["samples.json", "metadata.json", "fiftyone.yml", "data/*"],
                      max_workers=32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.skip_download:
        download(args.root)
    out = fiftyone_to_bdd(args.root)
    entries = json.loads(out.read_text(encoding="utf-8"))
    n_night = sum(1 for e in entries if e["attributes"]["timeofday"] == "night")
    print(f"변환 완료: {out} (전체 {len(entries)}장, 야간 {n_night}장)")


if __name__ == "__main__":
    main()
