"""Night-Enhancement (Jin et al., ECCV 2022, "Unsupervised Night Image Enhancement").
공식 저장소 jinyeying/night-enhancement의 생성기를 그대로 쓴다.

공식 ENHANCENET.py test()와 main_delighteffects.py 기본값에 맞춘 내용:
- 생성기: networks.ResnetGenerator(input_nc=3, output_nc=3, ngf=64(--ch), n_blocks=4(--n_res), img_size=512)
- 가중치: results/<dataset>/model/<dataset>_params_<step>.pt 의 params['genA2B']
  (delighteffects: 0600000, LOL: 0900000)
- 입력: PIL RGB → Resize((512, 512)) → ToTensor → Normalize(mean 0.5, std 0.5) → [-1,1]
- 출력: genA2B(real_A)가 (out, c_logit, heatmap)을 돌려주고 첫 번째가 결과.
  denorm(x*0.5+0.5) → 원래 (너비, 높이)로 cv2.resize → ×255
- 가중치는 README의 Dropbox 링크에서 사용자가 직접 받는다
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import EnhancerUnavailable
from .third_party import build_in_repo
from .torch_common import TorchRGBEnhancer, check_variant, require_file, require_torch, torch

VARIANTS = ("delighteffects", "lol")
_CKPT = {"delighteffects": ("delighteffects", "delighteffects_params_0600000.pt"),
         "lol": ("LOL", "LOL_params_0900000.pt")}
IMG_SIZE = 512
CH, N_RES = 64, 4


def weights_path(repo_dir: Path, variant: str) -> Path:
    folder, fname = _CKPT[variant]
    return Path(repo_dir) / "results" / folder / "model" / fname


class NightEnhEnhancer(TorchRGBEnhancer):
    name = "nightenh"

    def __init__(self, repo_dir: Path, variant: str = "delighteffects", device=None):
        check_variant(variant, VARIANTS, "Night-Enhancement")
        repo_dir = Path(repo_dir)
        if not repo_dir.is_dir():
            raise EnhancerUnavailable(f"외부 저장소 폴더 없음: {repo_dir} (python -m scripts.fetch_third_party)")
        weights = require_file(weights_path(repo_dir, variant), "README의 Dropbox 링크에서 받아 이 위치에 둔다")
        require_torch("Night-Enhancement")
        self._set_device(device)
        self.params = {"variant": variant}
        self.net = build_in_repo(repo_dir, "networks", "ResnetGenerator", input_nc=3, output_nc=3,
                                 ngf=CH, n_blocks=N_RES, img_size=IMG_SIZE).to(self.device).eval()
        params = torch.load(str(weights), map_location=self.device)
        if "genA2B" not in params:
            raise EnhancerUnavailable(f"{weights.name}에 genA2B 키가 없음 (키: {list(params)[:5]})")
        try:
            self.net.load_state_dict(params["genA2B"], strict=True)
        except RuntimeError as e:
            raise EnhancerUnavailable(f"가중치 구조가 Night-Enhancement 생성기와 맞지 않음: {e}") from e

    def preprocess(self, img: np.ndarray):
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def infer(self, x):
        with torch.no_grad():
            return self.net(x)[0]

    def postprocess(self, y, img_shape: tuple[int, int]) -> np.ndarray:
        h, w = img_shape
        out = (y[0] * 0.5 + 0.5).permute(1, 2, 0).float().cpu().numpy()
        out = cv2.resize(out, (w, h))
        out = (np.clip(out, 0, 1) * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
