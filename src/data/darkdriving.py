"""DarkDriving 로더 (github.com/DriveMindLab/DarkDriving-ICRA-2026, 학술·비상업 용도).

실제 받은 데이터 형식 (darkdriving_det.zip → Det_data/):
- 야간 이미지: night/images/<split>/<이름>.jpg, 2448×2048
- 라벨: night/labels_me/<split>/<이름>.json (labelme, shape_type rectangle, 점 2개 = 두 꼭짓점, 픽셀 좌표)
        night/labels/<split>/<이름>.txt (YOLO 정규화 cx cy w h, 클래스 0 = car) — 같은 박스. labelme를 쓴다
- 정렬된 낮 이미지: day/images/<split>/<같은 이름>.jpg → Sample.ref_path (정답 기반 화질 지표용)
- val: 야간 2,217장, 박스 3,564개(전부 car), 2,217장 모두 낮 짝이 있음
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import DetectionDataset, Sample, make_sample
from .classes import DARKDRIVING_TO_COMMON


class DarkDrivingDataset(DetectionDataset):
    name = "darkdriving"
    eval_classes = [2]   # car만 라벨이 있다

    def __init__(self, root: Path, split: str = "val"):
        self.root = Path(root)
        self.split = split

    def load(self) -> list[Sample]:
        label_dir = self.root / "night" / "labels_me" / self.split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"DarkDriving 라벨 폴더 없음: {label_dir}")
        img_dir = self.root / "night" / "images" / self.split
        day_dir = self.root / "day" / "images" / self.split
        samples = []
        for p in sorted(label_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            boxes, labels = [], []
            for s in d.get("shapes", []):
                cls = DARKDRIVING_TO_COMMON.get(s.get("label"))
                if cls is None or s.get("shape_type") != "rectangle":
                    continue
                (x1, y1), (x2, y2) = s["points"][:2]
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                labels.append(cls)
            name = Path(str(d.get("imagePath", "")).replace("\\", "/")).name or f"{p.stem}.jpg"
            ref = day_dir / name
            samples.append(make_sample(p.stem, img_dir / name, boxes, labels, ref_path=ref if ref.exists() else None))
        return samples
