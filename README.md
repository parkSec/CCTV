# 야간 도로 영상 화질 개선 × 객체 탐지 비교

어두운 도로 영상을 먼저 밝게 만들면 객체 탐지가 더 잘 되는지 실험으로 확인하는 프로젝트입니다.
화질 개선 조건 8개(원본 포함)를 같은 이미지, 같은 탐지기로 비교하고, 처리 시간·탐지 정확도·화질 지표를 함께 잽니다.

| 문서 | 내용 |
|---|---|
| **README.md** (이 문서) | 프로젝트 소개, 설치와 실행 방법 |
| [**RESULTS_enhancement.md**](RESULTS_enhancement.md) | 현재 실험(개선 모델 7개 × 데이터셋 4개) 결과, 분석, 한계 |
| [**RESULTS_traditional.md**](RESULTS_traditional.md) | 전통 기법 8개 비교와 파라미터 비교 결과 |
| [RESULTS.md](RESULTS.md) · [RESULTS_summary.pdf](RESULTS_summary.pdf) | 1차 실험(COCO YOLO26s, 데이터셋 2개) 기록 |
| [docs/](docs/) | 설계 문서, 구현 계획 |

**목차**
1. [프로젝트 소개](#1-프로젝트-소개)
2. [결과 요약](#2-결과-요약)
3. [사용 방법](#3-사용-방법)
4. [참고](#4-참고)

---

## 1. 프로젝트 소개

### 비교하는 조건 (8개)

| 조건 | 종류 | 설정 |
|---|---|---|
| 원본 | 기준 | 개선 없음 |
| Gamma | 전통 기법 | γ = 1.5 |
| CLAHE | 전통 기법 | LAB의 L 채널, clip 2.0, tile 8×8 |
| Zero-DCE | 딥러닝 (CVPR 2020) | 공식 가중치 Epoch99 |
| Zero-DCE++ | 딥러닝 (TPAMI 2021) | 공식 가중치, scale 1 |
| SCI | 딥러닝 (CVPR 2022) | difficult 가중치 |
| RUAS | 딥러닝 (CVPR 2021) | upe 가중치 (3개 중 비교해 고름) |
| Night-Enhancement | 딥러닝 (ECCV 2022) | delighteffects 가중치 (빛 번짐 억제) |

딥러닝 모델은 공식 저장소를 고정 commit으로 받아 모델 클래스와 공식 가중치를 그대로 씁니다. 이 프로젝트에서 새로 학습시킨 모델은 없습니다.

### 측정하는 것

- **탐지:** Precision, Recall, F1(신뢰도 0.25, IoU 0.5), AP50, AP50-95, 객체 크기별 AP
- **화질 (pyiqa):** 정답 기반 PSNR·SSIM·LPIPS(DarkDriving만), 정답 없는 NIQE·BRISQUE·MUSIQ·CLIP-IQA
- **처리 시간:** 개선을 준비·계산·후처리로 나눠 측정, 탐지 시간 따로, GPU 메모리
- **선택:** 실시간 기준(개선+탐지 33.3ms) 안의 "빠른 모델"과, 데이터셋 순위가 가장 안정적인 "확실한 모델"

### 탐지 모델 (측정 도구)

BDD100K로 학습된 YOLO26s·YOLO26m(Hugging Face `dronefreak/bdd100k-yolo26s`, `-yolo26m`)을 입력 크기 640·960으로 씁니다(탐지 설정 4개, 기준은 yolo26s 640). 탐지기는 다시 학습하지 않습니다.

### 데이터셋 (4개, 각 300장)

| 데이터셋 | 사용한 부분 | 평가 클래스 | 정렬된 낮 이미지 |
|---|---|---|---|
| BDD100K | 공식 val 중 야간 3,929장 | 6개 | 없음 |
| ExDark | 차량·사람 계열 4,077장 | 5개 (truck 없음) | 없음 |
| ACDC | night train·val 506장 | 6개 | 없음 |
| DarkDriving | 탐지용 val 2,217장 | car | 있음 |

- 클래스는 person, bicycle, car, motorcycle, bus, truck 6개로 맞춥니다.
- 데이터셋마다 시드 42로 300장을 뽑아 **모든 조건에 같은 이미지**를 씁니다. 촬영 환경이 달라 데이터셋은 합치지 않고 따로 비교합니다.
- 모든 이미지는 긴 변을 1280px 이하로 줄입니다(정답 박스도 같은 비율).

---

## 2. 결과 요약

> 전체 표와 분석은 [RESULTS_enhancement.md](RESULTS_enhancement.md)에 있습니다. (실험 번호 `20261117-230804`)

**실시간 기준 안에 들면서 탐지를 확실히 올리는 개선 방법은 없었습니다.**

| 조건 | 평균 ΔAP50 (4개 데이터셋, yolo26s 640) | 개선+탐지 (ms/장) | 33.3ms 이내 |
|---|---:|---:|---|
| Gamma (γ1.5) | +0.0006 | 14.6 | 예 |
| CLAHE (clip 2.0) | +0.0045 | 25.9 | 예 |
| Zero-DCE++ | −0.022 | 59.3 | 아니오 |
| Zero-DCE | −0.041 | 60.2 | 아니오 |
| SCI | −0.041 | 36.0 | 아니오 |
| Night-Enhancement | −0.044 | 97.5 | 아니오 |
| RUAS (upe) | −0.045 | 70.5 | 아니오 |

- 선택 규칙대로 고르면 **빠른 모델은 CLAHE, 확실한 모델은 Gamma**입니다. 하지만 두 기법 모두 평균 효과는 0에 가깝습니다.
- 딥러닝 모델 5개는 탐지 설정 4개 모두에서 평균 AP50이 떨어졌습니다.
- 화질 지표가 좋아져도 탐지가 좋아지지는 않았습니다. 지표와 탐지 변화의 상관은 데이터셋마다 방향이 달랐습니다.

### 전통 기법 비교

> 전체 표와 분석은 [RESULTS_traditional.md](RESULTS_traditional.md)에 있습니다. (실험 번호 `20261127-115620`)

전통 기법 8개를 같은 조건으로 비교한 결과입니다(평균 ΔAP50, yolo26s 640).

| 기법 | 평균 ΔAP50 | 개선 시간 (장당) |
|---|---:|---:|
| CLAHE (clip 2.0) | +0.0045 | 약 10~14ms |
| Gamma (γ1.5) | +0.0006 | 약 1ms |
| BIMEF | −0.002 | 약 1.1초 |
| AGCWD (α0.5) | −0.005 | 약 5~7ms |
| Dong | −0.049 | 약 0.19초 |
| MSRCR | −0.070 | 약 0.21초 |
| LIME | −0.075 | 약 2.4초 |
| HE | −0.088 | 약 11ms |

- 많이 밝히는 기법(HE, LIME, MSRCR, Dong)은 어두운 부분의 잡음까지 키워서 탐지가 떨어졌습니다. 화질 점수(NIQE, PSNR)는 오히려 가장 좋았습니다.
- 파라미터를 바꿔 보니 보정을 약하게 할수록 나았습니다. **Gamma γ1.2(+0.0072)**와 **CLAHE clip 1.0(+0.0075)**이 가장 좋았고, 이 둘을 각각 확실한 모델과 빠른 모델로 추천합니다. 다만 효과는 AP50 +0.01 미만입니다.

---

## 3. 사용 방법

### 3-1. 설치

**Python 3.11 또는 3.12**를 권장합니다.

```bash
# 1. 가상환경 (Git Bash 기준, PowerShell이면 .venv\Scripts\activate)
py -3.11 -m venv .venv
source .venv/Scripts/activate

# 2. PyTorch (GPU 버전) — 반드시 먼저 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 나머지 라이브러리 (pyiqa 포함)
pip install -r requirements.txt

# 4. GPU 인식 확인 → True 가 나오면 OK
python -c "import torch; print(torch.cuda.is_available())"
```

### 3-2. 데이터 받기

모두 `datasets/` 아래에 둡니다. 경로가 다르면 `configs/default.yaml`만 고치면 됩니다.

| 데이터셋 | 받는 방법 | 위치 |
|---|---|---|
| BDD100K | `python -m scripts.download_bdd100k_hf` (약 650MB) | `datasets/bdd100k_hf/` |
| ExDark | `python -m scripts.download_exdark` (약 1.5GB) | `datasets/ExDark/ExDark/`, `datasets/ExDark/ExDark_Annno/` |
| ACDC | [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch)에서 `gt_detection_trainval.zip`, `rgb_anon_trainvaltest.zip`을 받아 풂 | `datasets/ACDC/` (`gt_detection/`, `rgb_anon/`) |
| DarkDriving | [DarkDriving-ICRA-2026](https://github.com/DriveMindLab/DarkDriving-ICRA-2026) 안내에 따라 `darkdriving_det.zip`을 받아 풂 | `datasets/DarkDriving/Det_data/` |

ExDark는 Google Drive 다운로드 횟수 제한으로 실패하면, [공식 GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) 안내의 링크에서 zip 두 개를 직접 받은 뒤 이렇게 실행합니다.
```bash
python -m scripts.download_exdark --images-zip 이미지.zip --annos-zip 라벨.zip
```

데이터셋 폴더가 없으면 실험에서 그 데이터셋만 건너뛰고, 이유가 결과 파일에 남습니다.

### 3-3. 모델 받기

자세한 내용은 [scripts/download_weights.md](scripts/download_weights.md)에 있습니다.

```bash
# 탐지 모델 (BDD100K로 학습한 YOLO26s·m) → weights/
python -m scripts.download_detectors

# 외부 개선 모델 공식 저장소 (Zero-DCE, SCI, RUAS, Night-Enhancement) → third_party/
python -m scripts.fetch_third_party
```

- Zero-DCE++ 가중치 `weights/Epoch99.pth`는 [Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension) 저장소의 `Zero-DCE++/snapshots_Zero_DCE++/`에서 받습니다.
- Night-Enhancement 가중치는 `fetch_third_party`가 출력하는 위치에 Dropbox 링크(저장소 README)에서 직접 받아 둡니다.

### 3-4. 실행

```bash
# 테스트
python -m pytest -q

# 준비 상태 확인 (데이터셋·가중치·pyiqa·CUDA). 실험은 하지 않음
python -m scripts.run_experiment --check

# 작은 실행으로 흐름 확인 → 전체 실행 (데이터셋마다 300장, RTX 3070 기준 약 50분)
python -m scripts.run_experiment --num-samples 5
python -m scripts.run_experiment

# 일부 데이터셋만
python -m scripts.run_experiment --datasets bdd100k exdark

# 분석 (변화량, 화질-탐지 상관, 빠른/확실한 모델, 그림)
python -m scripts.analyze results/<run_id>

# RUAS 가중치 3개(lol, upe, dark) 비교 실험
python -m scripts.run_experiment --config configs/ruas_variants.yaml

# 전통 기법 8개 비교 (약 2시간 40분, LIME·BIMEF가 대부분), 감마·CLAHE·AGCWD 파라미터 비교
python -m scripts.run_experiment --config configs/traditional.yaml
python -m scripts.run_experiment --config configs/traditional_sweep.yaml
```

### 3-5. 결과 파일

실행할 때마다 `results/<날짜-시간>/` 폴더가 하나씩 생깁니다.

| 파일 | 내용 |
|---|---|
| `detection_metrics.csv` | 데이터셋 × 조건 × 탐지 설정(모델, 입력 크기) × 클래스 × 객체 크기별 P / R / F1 / AP50 / AP50-95 |
| `iqa_metrics.csv` | 데이터셋 × 조건별 화질 지표 평균·중앙값·NaN 개수 |
| `timing.csv` | 개선 단계별(준비·계산·후처리·합계) 시간 평균·중앙값·p95, 탐지 설정별 시간, GPU 최대 메모리 |
| `per_image.jsonl` | 이미지 한 장씩의 상세 결과 |
| `run_meta.json` | 실험 설정, 실행 환경, 외부 저장소 commit, 건너뛴 조건·데이터셋과 이유 |
| `samples/` | 조건별 처리 결과 이미지 |
| `analysis/` | `scripts.analyze` 결과: `delta.csv`, `iqa_correlation.csv`, `selection.json`, 그림 2개 |

### 3-6. 결과 화면과 비교 이미지

```bash
# 웹 화면: 결과 표·그래프, 저장된 비교 이미지, 직접 올린 이미지·영상 프레임 데모
streamlit run app/streamlit_app.py

# 영상의 한 프레임(또는 이미지 한 장)을 조건 8개로 개선·탐지해 비교 격자 이미지로 저장
python -m scripts.compare_frame --input 영상.mp4 --frame 120 --out results/compare.png
```

---

## 4. 참고

### 폴더 구조

```
CCTV/
├─ README.md                  # 소개, 설치·실행 방법 (이 문서)
├─ RESULTS_enhancement.md     # 현재 실험 결과와 분석
├─ RESULTS_traditional.md     # 전통 기법 비교 결과
├─ RESULTS.md                 # 1차 실험 기록
├─ configs/
│   ├─ default.yaml           # 실험 조건 (데이터셋 경로, 표본 수, 탐지 설정, 개선 조건)
│   ├─ ruas_variants.yaml     # RUAS 가중치 비교 (default.yaml을 물려받음)
│   ├─ traditional.yaml       # 전통 기법 8개 비교
│   └─ traditional_sweep.yaml # 감마·CLAHE·AGCWD 파라미터 비교
├─ src/
│   ├─ data/                  # BDD100K, ExDark, ACDC, DarkDriving 불러오기 + 클래스 통일, 리사이즈
│   ├─ enhance/               # Gamma, CLAHE, 전통 기법(HE·AGCWD·MSRCR·LIME·Dong·BIMEF), 딥러닝 모델 5개
│   ├─ detect/                # YOLO26
│   ├─ eval/                  # 박스 매칭, P/R/F1·AP, 화질 지표, 분석
│   ├─ timing.py              # 시간 측정
│   └─ io_utils.py            # 결과 저장 (사용자 경로 가림)
├─ scripts/
│   ├─ run_experiment.py      # 실험 실행
│   ├─ analyze.py             # 결과 분석
│   ├─ compare_frame.py       # 비교 격자 이미지
│   ├─ download_*.py          # 데이터셋·탐지 모델 받기
│   ├─ fetch_third_party.py   # 외부 개선 모델 저장소 받기
│   └─ download_weights.md    # 가중치 받는 방법
├─ app/                       # Streamlit 화면
├─ tests/                     # 테스트
└─ docs/                      # 설계 문서, 구현 계획
```

`datasets/`, `weights/`, `third_party/`, `results/`는 저장소에 올리지 않습니다.

### 출처와 라이선스

- **개선 모델:** Zero-DCE(Guo et al., CVPR 2020), Zero-DCE++(Li et al., TPAMI 2021), SCI(Ma et al., CVPR 2022), RUAS(Liu et al., CVPR 2021), Night-Enhancement(Jin et al., ECCV 2022). 각 공식 저장소의 라이선스를 따르며, Zero-DCE 계열은 비상업적 연구 목적으로만 사용할 수 있습니다.
- **전통 기법:** HE, CLAHE, AGCWD(Huang et al., TIP 2013), MSRCR(Jobson et al., TIP 1997), LIME(Guo et al., TIP 2017), Dong(ICME 2011), BIMEF(Ying et al., 2017). 논문 식을 보고 직접 구현했습니다(`src/enhance/traditional.py`).
- **데이터셋:** BDD100K(Yu et al., CVPR 2020, Hugging Face 재배포본 `dgural/bdd100k`), ExDark(Loh & Chan, CVIU 2019), ACDC(Sakaridis et al., ICCV 2021, 비상업 연구용), DarkDriving(Wang et al., ICRA 2026, 학술·비상업 용도).
- **탐지 모델:** Ultralytics YOLO26 (AGPL-3.0), BDD100K 학습 가중치 `dronefreak/bdd100k-yolo26s`, `-yolo26m`.
- **화질 지표:** pyiqa (chaofengc/IQA-PyTorch).
