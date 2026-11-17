"""Zero-DCE 원본 (Guo et al., CVPR 2020). 공식 저장소 Li-Chongyi/Zero-DCE의 모델 클래스를 그대로 쓴다.

공식 Zero-DCE_code/lowlight_test.py와 같게 맞춘 내용:
- 네트워크: model.enhance_net_nopool(), 가중치 snapshots/Epoch99.pth (load_state_dict)
- 입력: RGB, /255 → [0,1], (1,3,H,W). 평균·표준편차 정규화 없음
- 출력: forward가 (enhance_image_1, enhance_image, r)를 돌려주고 두 번째가 결과
- forward에 풀링·다운샘플이 없어 어떤 크기든 그대로 들어간다
Zero-DCE++(weights/Epoch99.pth)와 파일 이름이 같지만 다른 파일이다.
"""
from __future__ import annotations

from pathlib import Path

from .third_party import FETCH_HINT, build_in_repo
from .torch_common import TorchRGBEnhancer, load_checkpoint, require_file, require_torch, torch


class ZeroDCEOrigEnhancer(TorchRGBEnhancer):
    name = "zerodce"

    def __init__(self, repo_dir: Path, weights: Path, device=None):
        code_dir = Path(repo_dir) / "Zero-DCE_code"
        weights = require_file(weights, FETCH_HINT)
        require_torch("Zero-DCE")
        self._set_device(device)
        self.params = {"weights": Path(weights).name}
        self.net = build_in_repo(code_dir, "model", "enhance_net_nopool").to(self.device).eval()
        try:
            self.net.load_state_dict(load_checkpoint(weights, self.device), strict=True)
        except RuntimeError as e:
            from .base import EnhancerUnavailable
            raise EnhancerUnavailable(f"가중치 구조가 Zero-DCE와 맞지 않음: {e}") from e

    def infer(self, x):
        with torch.no_grad():
            return self.net(x)[1]
