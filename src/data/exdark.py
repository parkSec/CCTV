"""ExDark 로더. 구조: images_dir/<Class>/<img>, annos_dir/<Class>/<img>.txt (bbGt 형식)."""
from __future__ import annotations

from pathlib import Path

from .base import DetectionDataset, Sample, make_sample
from .classes import EXDARK_TO_COMMON

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_bbgt(text: str) -> list[tuple[str, float, float, float, float]]:
    """'Class l t w h ...' 줄들을 읽는다. '%'로 시작하는 줄과 빈 줄은 무시."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out


class ExDarkDataset(DetectionDataset):
    name = "exdark"
    eval_classes = [0, 1, 2, 3, 4]

    def __init__(self, images_dir: Path, annos_dir: Path):
        self.images_dir = Path(images_dir)
        self.annos_dir = Path(annos_dir)

    def _anno_index(self, cls_dir: str) -> dict[str, Path]:
        """라벨 파일을 소문자 이름으로 색인 (확장자 대소문자 차이 흡수)."""
        d = self.annos_dir / cls_dir
        if not d.is_dir():
            return {}
        return {p.name.lower(): p for p in d.iterdir() if p.is_file()}

    def load(self) -> list[Sample]:
        samples = []
        for cls_dir in sorted(p for p in self.images_dir.iterdir() if p.is_dir()):
            annos = self._anno_index(cls_dir.name)
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                anno = annos.get((img.name + ".txt").lower())
                if anno is None:
                    continue
                boxes, labels = [], []
                for name, l, t, w, h in parse_bbgt(anno.read_text(encoding="utf-8", errors="ignore")):
                    cls = EXDARK_TO_COMMON.get(name)
                    if cls is None:
                        continue
                    boxes.append([l, t, l + w, t + h])
                    labels.append(cls)
                if not labels:
                    continue
                samples.append(make_sample(f"{cls_dir.name}/{img.name}", img, boxes, labels))
        return samples
