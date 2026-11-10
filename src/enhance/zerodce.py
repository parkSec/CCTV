"""Zero-DCE++ (Li et al., TPAMI 2021).

공식 구현(Li-Chongyi/Zero-DCE_extension, Zero-DCE++/model.py)과 같은 모듈 이름을 써야
공식 사전학습 가중치(Epoch99.pth)가 그대로 로드된다.
PyTorch가 없어도 이 파일을 import할 수 있어야 하므로 torch는 선택적으로 불러온다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Enhancer, EnhancerUnavailable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # GPU 없는 작업 환경
    torch = None


def pad_to_multiple(img: np.ndarray, k: int) -> tuple[np.ndarray, tuple[int, int]]:
    """H, W를 k의 배수로 맞추도록 아래·오른쪽을 가장자리 값으로 채운다. (패딩된 이미지, 원래 (h, w))."""
    h, w = img.shape[:2]
    ph, pw = (-h) % k, (-w) % k
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return img, (h, w)


if torch is not None:

    class CSDN_Tem(nn.Module):
        """depthwise + pointwise 합성곱."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
            self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1)

        def forward(self, x):
            return self.point_conv(self.depth_conv(x))

    class ZeroDCEPPNet(nn.Module):
        """공식 enhance_net_nopool (Zero-DCE++)."""

        def __init__(self, scale_factor: int = 1):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.scale_factor = scale_factor
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            nf = 32
            self.e_conv1 = CSDN_Tem(3, nf)
            self.e_conv2 = CSDN_Tem(nf, nf)
            self.e_conv3 = CSDN_Tem(nf, nf)
            self.e_conv4 = CSDN_Tem(nf, nf)
            self.e_conv5 = CSDN_Tem(nf * 2, nf)
            self.e_conv6 = CSDN_Tem(nf * 2, nf)
            self.e_conv7 = CSDN_Tem(nf * 2, 3)

        @staticmethod
        def curve(x, x_r):
            for _ in range(8):
                x = x + x_r * (torch.pow(x, 2) - x)
            return x

        def forward(self, x):
            x_down = x if self.scale_factor == 1 else F.interpolate(
                x, scale_factor=1 / self.scale_factor, mode="bilinear")
            x1 = self.relu(self.e_conv1(x_down))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
            if self.scale_factor != 1:
                x_r = self.upsample(x_r)
            return self.curve(x, x_r), x_r


def _load_state(path: Path, device):
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k.removeprefix("module."): v for k, v in state.items()}


class ZeroDCEEnhancer(Enhancer):
    name = "zerodce"

    def __init__(self, weights: Path, scale_factor: int = 1, device: str | None = None):
        weights = Path(weights)
        if not weights.exists():
            raise EnhancerUnavailable(f"가중치 파일 없음: {weights}")
        if torch is None:
            raise EnhancerUnavailable("torch가 설치되어 있지 않음")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.uses_cuda = self.device.type == "cuda"
        self.scale_factor = int(scale_factor)
        self.params = {"scale_factor": self.scale_factor, "weights": weights.name}
        self.net = ZeroDCEPPNet(self.scale_factor).to(self.device).eval()
        try:
            self.net.load_state_dict(_load_state(weights, self.device), strict=True)
        except RuntimeError as e:
            raise EnhancerUnavailable(f"가중치 구조가 Zero-DCE++와 맞지 않음: {e}") from e

    def enhance(self, img: np.ndarray) -> np.ndarray:
        padded, (h, w) = pad_to_multiple(img, self.scale_factor)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y, _ = self.net(x)
        y = (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)[:h, :w]
