# 저조도 도로 영상 화질 개선 × 객체 탐지 비교

야간·저조도 도로 영상은 차량과 보행자가 잘 보이지 않습니다.
그렇다면 **영상을 밝고 선명하게 만들면 객체 탐지도 실제로 더 잘 될까?**
이 프로젝트는 여러 화질 개선 기법을 같은 조건에서 적용하고, 같은 탐지 모델(YOLO26)로 결과를 비교해서 이 질문에 답해 봅니다.

- 정확도: Precision, Recall, F1, mAP@0.5
- 속도: 화질 개선 시간과 탐지 시간을 **따로** 측정
- 결과: CSV / JSON 파일과 비교 이미지로 저장

---

## 비교하는 기법

| 기법 | 종류 | 한 줄 설명 |
|---|---|---|
| 원본 | 기준선 | 아무 처리도 하지 않은 이미지 |
| Gamma 보정 | 전통 기법 | 어두운 영역을 밝게 펴 줌 (γ = 1.5 / 2.0 / 2.5) |
| CLAHE | 전통 기법 | 영역별로 대비를 높임, 밝기 채널에만 적용 (clip = 2.0 / 3.0 / 4.0) |
| Zero-DCE++ | 딥러닝 | 저조도 개선 전용 경량 모델 (파라미터 약 1만 개) |
| Real-ESRGAN | 딥러닝 | 노이즈·블러 복원 모델. 2배로 키운 뒤 원래 크기로 되돌려 사용 |

총 **9개 조건**(원본 1 + Gamma 3 + CLAHE 3 + Zero-DCE++ 1 + Real-ESRGAN 1)을 비교합니다.

## 공정한 비교를 위해 고정한 것

- 탐지 모델: YOLO26s (사전학습), 입력 크기 640
- 신뢰도 기준: 0.25 (P/R/F1), AP는 전체 점수 범위
- 정답 판정: 같은 클래스 + IoU 0.5 이상
- 이미지: 데이터셋마다 300장을 시드 42로 뽑아 **모든 기법에 같은 이미지** 사용
- 평가 클래스: person, bicycle, car, motorcycle, bus, truck
- 데이터셋 결과는 **합치지 않고 따로** 비교 (촬영 환경이 달라서)

## 데이터셋

| 데이터셋 | 용도 | 비고 |
|---|---|---|
| BDD100K 야간 | 메인 평가 | 검증 세트 중 야간 3,929장 |
| ExDark | 일반화 확인 | 저조도 이미지, 차량·사람 계열 클래스만 사용 |

---

## 설치

**Python 3.11 또는 3.12**가 필요합니다. (Real-ESRGAN이 쓰는 `basicsr`가 3.13 이상에서는 설치되지 않습니다.)

```bash
# 1. 가상환경 (Git Bash 기준, PowerShell이면 .venv\Scripts\activate)
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version            # 3.11.x 확인

# 2. PyTorch (GPU 버전) — 반드시 먼저 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 나머지 라이브러리
pip install -r requirements.txt

# 4. GPU 인식 확인 → True 가 나오면 OK
python -c "import torch; print(torch.cuda.is_available())"
```

## 준비물

### 1) 데이터

**BDD100K** — 명령 한 줄로 받고 변환까지 끝납니다. (약 650MB, 20분 정도)
```bash
python -m scripts.download_bdd100k_hf
```
끝나면 `datasets/bdd100k_hf/` 폴더가 생깁니다.

**ExDark** — [공식 GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)에서 받아 아래처럼 둡니다.
```
datasets/ExDark/
├─ ExDark/<클래스>/<이미지>
└─ ExDark_Annno/<클래스>/<이미지 파일명>.txt
```

> 경로가 다르면 `configs/default.yaml`만 고치면 됩니다.

### 2) 모델 가중치

`weights/` 폴더에 넣습니다. 자세한 내용은 `scripts/download_weights.md` 참고.

| 파일 | 기법 | 받는 곳 |
|---|---|---|
| `yolo26s.pt` | YOLO26 | 처음 실행할 때 자동으로 받아짐 |
| `Epoch99.pth` | Zero-DCE++ | [Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension) 저장소의 `Zero-DCE++/snapshots_Zero_DCE++/` |
| `RealESRGAN_x2plus.pth` | Real-ESRGAN | 아래 명령 |

```bash
curl -L -o weights/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

가중치가 없는 기법은 실험에서 자동으로 건너뛰고, 이유가 결과 파일에 남습니다.

---

## 실행

```bash
# 테스트 (설치가 잘 됐는지 확인)
pytest -rs

# 빠른 확인: 데이터셋마다 20장만
python -m scripts.run_experiment --num-samples 20

# BDD100K만 돌리기 (ExDark를 아직 안 받았을 때)
python -m scripts.run_experiment --num-samples 20 --datasets bdd100k

# 전체 실험: 데이터셋마다 300장 × 9개 조건
python -m scripts.run_experiment
```

## 결과 보기

실행할 때마다 `results/<날짜-시간>/` 폴더가 하나씩 생깁니다.

| 파일 | 내용 |
|---|---|
| `metrics.csv` | 데이터셋 × 기법 × 클래스별 Precision / Recall / F1 / AP50. `class = all` 행이 전체 성능 |
| `timing.csv` | 기법별 화질 개선 시간, 탐지 시간 (평균·중앙값, ms/장) |
| `per_image.jsonl` | 이미지 한 장씩의 상세 결과 |
| `run_meta.json` | 실험 설정, 실행 환경(GPU 등), 건너뛴 기법과 이유 |
| `samples/` | 기법별 처리 결과 이미지 (빨강 = 정답 박스, 초록 = 탐지 결과) |

CSV 파일은 엑셀에서 바로 열 수 있습니다.

---

## 폴더 구조

```
CCTV/
├─ configs/default.yaml     # 실험 조건 (데이터셋 경로, 표본 수, 기법·파라미터)
├─ src/
│   ├─ data/                # BDD100K, ExDark 불러오기 + 클래스 통일
│   ├─ enhance/             # Gamma, CLAHE, Zero-DCE++, Real-ESRGAN
│   ├─ detect/              # YOLO26
│   ├─ eval/                # 박스 매칭, P/R/F1, AP 계산
│   ├─ timing.py            # 시간 측정
│   └─ io_utils.py          # 결과 저장
├─ scripts/
│   ├─ run_experiment.py    # 실험 실행
│   └─ download_bdd100k_hf.py
├─ tests/                   # 테스트
├─ docs/                    # 설계 문서, 구현 계획
└─ work_history.md          # 작업 기록
```

## 일정

| 주차 | 내용 |
|---|---|
| 11주차 | 데이터 불러오기, Gamma·CLAHE, YOLO26 기준선, 정확도 계산 |
| 12주차 | Zero-DCE++, Real-ESRGAN, 전체 실험 |
| 13주차 | 웹 화면(Streamlit), 비교 영상 만들기, (선택) YOLO26 야간 파인튜닝 |
| 14주차 | 결과 정리, 발표 준비 |

---

## 출처와 라이선스

- **BDD100K**: Yu et al., CVPR 2020. 교육·연구·비영리 목적으로 사용. 이 프로젝트는 Hugging Face 재배포본(`dgural/bdd100k`)의 검증 세트를 사용합니다.
- **ExDark**: Loh & Chan, CVIU 2019.
- **Zero-DCE++**: Li et al., TPAMI 2021. 비상업적 연구 목적으로만 사용 가능.
- **Real-ESRGAN**: Wang et al., ICCVW 2021.
- **YOLO26**: Ultralytics (AGPL-3.0).
