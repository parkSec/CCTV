"""SCI (Ma et al., CVPR 2022). 공식 저장소 vis-opt-group/SCI의 CVPR 폴더 모델을 그대로 쓴다.

공식 CVPR/test.py와 같게 맞춘 내용:
- 모델: model.Finetunemodel(가중치 경로). 가중치 CVPR/weights/{easy,medium,difficult}.pt
- 입력: RGB, ToTensor → [0,1], (1,3,H,W). 정규화 없음 (multi_read_data.MemoryFriendlyLoader test)
- 출력: forward가 (i, r)을 돌려주고, r = clamp(입력 / i, 0, 1)이 결과 (save_images(r))
주의: Finetunemodel은 가중치 키 중 모델에 있는 것만 골라 넣기 때문에, 키가 하나도 맞지 않아도 오류가
나지 않고 무작위 가중치로 동작한다. 그래서 래퍼가 enhance.* 키가 전부 채워졌는지 따로 확인한다.
"""
from __future__ import annotations

from pathlib import Path

from .base import EnhancerUnavailable
from .third_party import FETCH_HINT, build_in_repo
from .torch_common import TorchRGBEnhancer, check_variant, load_checkpoint, require_file, require_torch, torch

VARIANTS = ("easy", "medium", "difficult")


def missing_model_keys(model_state: dict, ckpt: dict) -> list[str]:
    """모델 키 중 가중치 파일에 없는 것."""
    return [k for k in model_state if k not in ckpt]


class SCIEnhancer(TorchRGBEnhancer):
    name = "sci"

    def __init__(self, repo_dir: Path, variant: str = "difficult", device=None):
        check_variant(variant, VARIANTS, "SCI")
        code_dir = Path(repo_dir) / "CVPR"
        weights = require_file(code_dir / "weights" / f"{variant}.pt", FETCH_HINT)
        require_torch("SCI")
        self._set_device(device)
        self.params = {"variant": variant}
        net = build_in_repo(code_dir, "model", "Finetunemodel", str(weights))
        enhance_keys = {k: v for k, v in net.state_dict().items() if k.startswith("enhance.")}
        missing = missing_model_keys(enhance_keys, load_checkpoint(weights, "cpu"))
        if not enhance_keys or missing:
            raise EnhancerUnavailable(f"SCI 가중치 {weights.name}에 모델 키가 없음: {missing[:5]}")
        self.net = net.to(self.device).eval()

    def infer(self, x):
        with torch.no_grad():
            return self.net(x)[1]
