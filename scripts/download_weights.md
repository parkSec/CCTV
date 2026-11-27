# 가중치 받기

경로는 `configs/default.yaml`과 같아야 합니다. 가중치가 없는 조건은 실험에서 건너뛰고 이유가 `run_meta.json`의 `skipped_conditions`에 남습니다. `python -m scripts.run_experiment --check`로 한 번에 확인할 수 있습니다.

## 탐지 모델

```bash
python -m scripts.download_detectors
```

| 파일 | 용도 | 받는 곳 |
|---|---|---|
| `weights/bdd100k-yolo26s.pt` | 탐지 (기준) | Hugging Face `dronefreak/bdd100k-yolo26s`의 `best.pt` |
| `weights/bdd100k-yolo26m.pt` | 탐지 | Hugging Face `dronefreak/bdd100k-yolo26m`의 `best.pt` |

## 화질 개선 모델

```bash
python -m scripts.fetch_third_party
```

공식 저장소 4개를 `third_party/` 아래에 고정 commit(`third_party_manifest.json`)으로 받습니다. Zero-DCE, SCI, RUAS는 가중치가 저장소에 들어 있어 따로 받을 것이 없습니다.

| 조건 | 가중치 위치 | 받는 곳 |
|---|---|---|
| Zero-DCE | `third_party/zerodce/Zero-DCE_code/snapshots/Epoch99.pth` | 저장소에 포함 |
| Zero-DCE++ | `weights/Epoch99.pth` | [Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension)의 `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`를 직접 받음 |
| SCI | `third_party/sci/CVPR/weights/{easy,medium,difficult}.pt` | 저장소에 포함 |
| RUAS | `third_party/ruas/ckpt/{lol,upe,dark}.pt` (+ `model/denoise.pt`) | 저장소에 포함 |
| Night-Enhancement | `third_party/nightenh/results/delighteffects/model/delighteffects_params_0600000.pt`<br>`third_party/nightenh/results/LOL/model/LOL_params_0900000.pt` | 저장소 README의 Dropbox 링크에서 직접 받음 (각 약 512MB) |

Zero-DCE와 Zero-DCE++의 가중치는 이름(`Epoch99.pth`)이 같지만 다른 파일입니다.

## 확인 방법

```bash
python -m pytest tests/test_detect.py tests/test_zerodce.py tests/test_zerodce_orig.py tests/test_sci.py tests/test_ruas.py tests/test_nightenh.py -rs
```
- `skipped`가 남아 있으면 그 이유(가중치 없음 / 저장소 없음)가 함께 출력됩니다.
- `test_zerodce.py`의 건너뜀 1개("torch가 설치된 환경에서는 해당 없음")는 정상입니다.

## 참고
- Zero-DCE / Zero-DCE++ 코드는 비상업적 연구 목적으로만 사용할 수 있습니다.
- Zero-DCE++ 논문의 기본 축소 배율은 12입니다. 이 프로젝트는 줄이지 않고 계산하도록 1을 씁니다(`configs/default.yaml`의 `scale_factor`).
- 탐지 모델은 Ultralytics YOLO26(AGPL-3.0) 기반입니다.
