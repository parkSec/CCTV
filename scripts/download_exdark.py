"""ExDark(공식 Google Drive)를 받아 압축을 풀고 실험 코드가 읽는 폴더 구조로 정리한다.

사용: python -m scripts.download_exdark
      python -m scripts.download_exdark --images-zip 받은이미지.zip --annos-zip 받은라벨.zip   (직접 받은 경우)

결과:
  datasets/ExDark/ExDark/<클래스>/<이미지>
  datasets/ExDark/ExDark_Annno/<클래스>/<이미지 파일명>.txt

출처: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset (이미지 약 1.5GB, 라벨 약 4.2MB)
이용 조건: 비상업적 학술 연구·교육 목적 (저장소의 DATASET_TERMS.md 참고)
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.config import PROJECT_ROOT

IMAGES_DRIVE_ID = "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"   # Dataset/README.md 링크
ANNOS_DRIVE_ID = "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"    # Groundtruth/README.md 링크
DEFAULT_DEST = PROJECT_ROOT / "datasets" / "ExDark"

CLASSES = {"Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_zip(zip_path: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)


def _class_dirs(d: Path) -> list[Path]:
    return [c for c in d.iterdir() if c.is_dir() and c.name in CLASSES] if d.is_dir() else []


def _is_kind(class_dir: Path, kind: str) -> bool:
    for f in class_dir.iterdir():
        if kind == "annos" and f.name.lower().endswith(".txt"):
            return True
        if kind == "images" and f.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def find_class_root(work: Path, kind: str) -> Path | None:
    """클래스 이름 폴더(Car, People, ...)들을 가장 많이 가진 폴더를 찾는다. kind: 'images' 또는 'annos'."""
    best, best_n = None, 0
    if not work.is_dir():
        return None
    for d in [work, *(p for p in work.rglob("*") if p.is_dir())]:
        n = sum(1 for c in _class_dirs(d) if _is_kind(c, kind))
        if n > best_n:
            best, best_n = d, n
    return best


def arrange_layout(work: Path, dest: Path) -> None:
    for kind, name in (("images", "ExDark"), ("annos", "ExDark_Annno")):
        target = dest / name
        if any(_is_kind(c, kind) for c in _class_dirs(target)):
            continue  # 이미 정리됨
        root = find_class_root(work, kind)
        if root is None:
            raise FileNotFoundError(f"압축을 푼 폴더에서 {kind} 클래스 폴더를 찾지 못했습니다: {work}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(target))


def download(file_id: str, out: Path) -> Path:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("gdown이 없습니다. `pip install -r requirements.txt`를 먼저 실행하세요.") from e
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"이미 받은 파일 사용: {out}")
        return out
    result = gdown.download(id=file_id, output=str(out), quiet=False)
    if not result:
        raise SystemExit(
            "Google Drive에서 받지 못했습니다(다운로드 횟수 제한일 수 있음).\n"
            "브라우저로 직접 받은 뒤 --images-zip / --annos-zip 옵션으로 경로를 넘겨 주세요.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--images-zip", type=Path)
    ap.add_argument("--annos-zip", type=Path)
    args = ap.parse_args(argv)

    dl = args.dest / "_downloads"
    images_zip = args.images_zip or download(IMAGES_DRIVE_ID, dl / "ExDark_images.zip")
    annos_zip = args.annos_zip or download(ANNOS_DRIVE_ID, dl / "ExDark_annotations.zip")
    work = args.dest / "_extract"
    for z in (images_zip, annos_zip):
        print(f"압축 해제: {z.name}")
        extract_zip(z, work)
    arrange_layout(work, args.dest)

    from src.data.exdark import ExDarkDataset
    n = len(ExDarkDataset(args.dest / "ExDark", args.dest / "ExDark_Annno").load())
    print(f"정리 완료: {args.dest} (차량·사람 계열 라벨이 있는 이미지 {n}장)")


if __name__ == "__main__":
    main()
