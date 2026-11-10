"""BDD100K 검출 라벨(JSON) 로더."""
from __future__ import annotations

import json
from pathlib import Path

from .base import DetectionDataset, Sample, make_sample
from .classes import BDD_TO_COMMON


class BDD100KDataset(DetectionDataset):
    name = "bdd100k"
    eval_classes = [0, 1, 2, 3, 4, 5]

    def __init__(self, labels_json: Path, images_dir: Path, timeofday: str = "night"):
        self.labels_json = Path(labels_json)
        self.images_dir = Path(images_dir)
        self.timeofday = timeofday

    def load(self) -> list[Sample]:
        with open(self.labels_json, encoding="utf-8") as f:
            entries = json.load(f)
        samples = []
        for e in entries:
            if (e.get("attributes") or {}).get("timeofday") != self.timeofday:
                continue
            boxes, labels = [], []
            for lab in e.get("labels") or []:
                cls = BDD_TO_COMMON.get(lab.get("category"))
                box = lab.get("box2d")
                if cls is None or not box:
                    continue
                boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                labels.append(cls)
            samples.append(make_sample(e["name"], self.images_dir / e["name"], boxes, labels))
        return samples
