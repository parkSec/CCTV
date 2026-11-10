# 저조도 도로 영상 화질 개선 × 객체 탐지 비교

컴퓨터비전 팀프로젝트 10조. 설계는 `docs/superpowers/specs/`, 구현 계획은 `docs/superpowers/plans/`, 작업 기록은 `work_history.md`를 보세요.

## 설치 (Windows, RTX 3070)

**Python 3.11 또는 3.12가 필요합니다.** Real-ESRGAN 공식 패키지가 쓰는 basicsr는 Python 3.13 이상에서 설치되지 않습니다.

```bash
# Git Bash 기준 (PowerShell이면 .venv\Scripts\activate)
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version          # 3.11.x 확인
# 1) PyTorch(CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) 나머지
pip install -r requirements.txt
```

## 데이터 배치

**BDD100K (기본: Hugging Face 사본, 가입 불필요)**
```bash
python -m scripts.download_bdd100k_hf
```
`datasets/bdd100k_hf/`에 검증 세트(약 700MB)를 받고 `bdd_labels.json`으로 변환합니다. 비공식 재배포본이며 BDD100K 라이선스(교육·연구·비영리)를 따릅니다.

공식 사이트에서 받았다면 아래처럼 두고 `configs/default.yaml`의 bdd100k 경로를 주석에 적힌 값으로 바꾸세요.

```
datasets/
├─ bdd100k_hf/               ← 스크립트가 만든 폴더 (기본)
│   ├─ data/*.jpg
│   └─ bdd_labels.json
├─ bdd100k/                  ← 공식 사이트에서 받은 경우
│   ├─ images/100k/val/*.jpg
│   └─ labels/det_20/det_val.json
└─ ExDark/
    ├─ ExDark/<Class>/<이미지>
    └─ ExDark_Annno/<Class>/<이미지 파일명>.txt
```
경로가 다르면 `configs/default.yaml`만 고치면 됩니다.

## 가중치

`weights/` 폴더에 둡니다. 받는 곳과 확인 방법은 `scripts/download_weights.md`를 보세요.
- `yolo26s.pt`: ultralytics가 처음 실행할 때 자동으로 받음
- `Epoch99.pth`: Zero-DCE++
- `RealESRGAN_x2plus.pth`: Real-ESRGAN

가중치가 없는 기법은 실험에서 건너뛰고, 이유가 `run_meta.json`의 `skipped_conditions`에 남습니다.

## 실행

```bash
pytest                                                    # 테스트
python -m scripts.run_experiment --config configs/default.yaml --num-samples 20   # 빠른 확인
python -m scripts.run_experiment --config configs/default.yaml                    # 전체 실험
```
결과는 `results/<run_id>/`에 `metrics.csv`, `timing.csv`, `per_image.jsonl`, `run_meta.json`, `samples/`로 저장됩니다.
