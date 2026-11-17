"""ACDC 야간 로더 (acdc.vision.ee.ethz.ch, 비상업 연구용).

실제 받은 데이터 형식 (gt_detection_trainval.zip, rgb_anon_trainvaltest.zip):
- 라벨: <root>/gt_detection/<condition>/instancesonly_<condition>_<split>_gt_detection.json (COCO 형식)
  images[].file_name(예: night/val/GOPR0351/GOPR0351_frame_000818_rgb_anon.png), annotations[].bbox = xywh,
  iscrowd, categories[].name (person, rider, car, truck, bus, train, motorcycle, bicycle)
- 이미지: <root>/rgb_anon/<file_name>, 1920×1080 PNG
- 야간 train 400장(박스 2,975), val 106장(박스 559). test는 정답이 없어 쓰지 않는다.
클래스는 이름으로 매핑하고, iscrowd = 1(무리)과 train(열차)은 버린다. 박스가 없는 이미지도 넣는다(오탐 계산용).
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import DetectionDataset, Sample, make_sample
from .classes import ACDC_TO_COMMON


class ACDCDataset(DetectionDataset):
    eval_classes = [0, 1, 2, 3, 4, 5]

    def __init__(self, root: Path, condition: str = "night", splits: tuple[str, ...] = ("train", "val")):
        self.root = Path(root)
        self.condition = condition
        self.splits = tuple(splits)
        self.name = f"acdc_{condition}"

    def _json(self, split: str) -> Path:
        return self.root / "gt_detection" / self.condition / f"instancesonly_{self.condition}_{split}_gt_detection.json"

    def load(self) -> list[Sample]:
        samples = []
        for split in self.splits:
            path = self._json(split)
            if not path.exists():
                raise FileNotFoundError(f"ACDC 라벨 없음 ({split}): {path}")
            d = json.loads(path.read_text(encoding="utf-8"))
            names = {c["id"]: c["name"] for c in d["categories"]}
            per_image: dict[int, tuple[list, list]] = {i["id"]: ([], []) for i in d["images"]}
            for a in d["annotations"]:
                cls = ACDC_TO_COMMON.get(names.get(a["category_id"]))
                if cls is None or a.get("iscrowd", 0):
                    continue
                x, y, w, h = a["bbox"]
                per_image[a["image_id"]][0].append([x, y, x + w, y + h])
                per_image[a["image_id"]][1].append(cls)
            for img in d["images"]:
                boxes, labels = per_image[img["id"]]
                fname = img["file_name"]
                image_id = fname.removesuffix("_rgb_anon.png").removesuffix(".png")
                samples.append(make_sample(image_id, self.root / "rgb_anon" / fname, boxes, labels))
        return samples
