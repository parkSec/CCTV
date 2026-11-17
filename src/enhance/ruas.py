"""RUAS (Liu et al., CVPR 2021). 공식 저장소 KarelZhang/RUAS의 모델 클래스를 그대로 쓴다.

공식 test.py와 같게 맞춘 내용:
- 모델: model.Network() (인자 없음), 가중치 ckpt/{lol,upe,dark}.pt (load_state_dict)
- 입력: multi_read_data.MemoryFriendlyLoader(task='test') — PIL로 RGB 변환, ToTensor → [0,1],
  크기 조정·정규화 없음, (1,3,H,W)
- 출력: forward가 (u_list, t_list)를 돌려준다. lol은 u_list[-1], upe·dark는 u_list[-2]를 저장
- 원래 PyTorch 0.4 코드: test.py의 Variable(volatile=True)는 래퍼에서 쓰지 않고 torch.no_grad()로 대신함.
  model.py는 Variable을 import만 하고 쓰지 않음
- 다운샘플이 없어 어떤 크기든 그대로 들어간다
"""
from __future__ import annotations

from pathlib import Path

from .base import EnhancerUnavailable
from .third_party import FETCH_HINT, build_in_repo
from .torch_common import TorchRGBEnhancer, check_variant, load_checkpoint, require_file, require_torch, torch

VARIANTS = ("lol", "upe", "dark")


def select_output(u_list, variant: str):
    """공식 test.py의 저장 규칙."""
    return u_list[-1] if variant == "lol" else u_list[-2]


class RUASEnhancer(TorchRGBEnhancer):
    name = "ruas"

    def __init__(self, repo_dir: Path, variant: str = "dark", device=None):
        check_variant(variant, VARIANTS, "RUAS")
        repo_dir = Path(repo_dir)
        weights = require_file(repo_dir / "ckpt" / f"{variant}.pt", FETCH_HINT)
        require_torch("RUAS")
        self._set_device(device)
        self.variant = variant
        self.params = {"variant": variant}
        self.net = build_in_repo(repo_dir, "model", "Network").to(self.device).eval()
        try:
            self.net.load_state_dict(load_checkpoint(weights, self.device), strict=True)
        except RuntimeError as e:
            raise EnhancerUnavailable(f"가중치 구조가 RUAS와 맞지 않음: {e}") from e

    def infer(self, x):
        with torch.no_grad():
            u_list, _ = self.net(x)
            return select_output(u_list, self.variant)
