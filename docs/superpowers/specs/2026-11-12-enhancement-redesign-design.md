# 야간 도로 영상 화질 개선 × 객체 탐지 비교 — 재설계 스펙

- 작성일: 2026-11-12
- 이전 스펙: `2026-11-10-lowlight-detection-design.md` (복원 모델 Real-ESRGAN 포함, 데이터셋 2개) → 이 문서로 대체
- 결정 기록: `work_history.md`의 "Phase R0" 이후 항목
- 상태: 설계 확정 (2026-11-12, PairLIE·CLIP-LIT 제외, 3.1은 B 방식)

---

## 1. 목표

야간에 주행 중인 차량 카메라로 찍은 **원래 화질이 낮은 영상**에서 프레임을 이미지로 가져와, 여러 **화질 개선(enhancement) 모델**을 적용한다.
그 결과를 **처리 성능 · 정확성(탐지) · 향상성(화질)** 세 축으로 비교하고, **빠른 모델과 성능이 확실한 모델**을 고른다.

### 이번 재설계에서 바뀐 점

| 항목 | 이전 | 지금 |
|---|---|---|
| 목표 | 복원 + 개선 | **개선만** (손상이 아니라 원래 화질이 낮은 영상) |
| 개선 모델 | Gamma, CLAHE, Zero-DCE++, Real-ESRGAN | Gamma, CLAHE, Zero-DCE, Zero-DCE++, SCI, RUAS, Night-Enhancement (7개) |
| 탐지 모델 | COCO YOLO26s | BDD100K로 학습한 YOLO26s·YOLO26m, 입력 640·960 |
| 데이터셋 | BDD100K 야간, ExDark | BDD100K 야간, DarkDriving, ACDC 야간, AI Hub 고정밀 야간, ExDark(보조) |
| 평가 | P/R/F1, AP50, 처리 시간 | + mAP@50-95, 객체 크기별, 화질 지표 7개, 지표-탐지 상관관계 |
| 결과물 | CSV/JSON (+ 계획: 비교 영상) | CSV/JSON + **비교 이미지** + 웹 화면. 비교 영상은 제외 |

### 성공 기준

1. 5개 데이터셋 × 8개 조건(원본 + 개선 7개) × 탐지 2개 × 입력 크기 2개의 탐지 지표가 CSV로 나온다.
2. 같은 조건의 화질 지표 7개가 CSV로 나온다 (정답 기반 3개는 DarkDriving만).
3. 개선 시간과 탐지 시간이 따로, 그리고 모델 추론 시간과 전처리 시간이 따로 기록된다.
4. 화질 지표와 탐지 성능 변화(ΔmAP)의 상관관계 표가 나온다.
5. "빠른 모델"과 "확실한 모델"이 규칙(7장)에 따라 자동으로 뽑힌다.
6. 영상 파일을 넣으면 지정한 프레임을 이미지로 뽑아 원본·개선본·탐지 결과를 나란히 비교한 이미지를 낸다.
7. 웹 화면에서 결과 표·그래프와 비교 이미지를 볼 수 있다.

### 범위 밖

- 비교 영상 내보내기, 영상 전용 개선 모델, 프레임 간 깜빡임(시간 일관성) 평가
- 개선 모델이나 탐지 모델의 재학습·미세조정
- COCO YOLO26s 기준선 (보류)

---

## 2. 확정된 구성

### 2.1 화질 개선 모델 (비교 대상)

- 제외(사용자 결정, 2026-11-12): PairLIE, CLIP-LIT

| 조건 이름 | 모델 | 논문 | 가중치 |
|---|---|---|---|
| `original` | 원본 그대로 | — | — |
| `gamma` | Gamma 보정 | 고전 기법 | 없음 |
| `clahe` | CLAHE (LAB의 L 채널) | Zuiderveld, Graphics Gems IV, 1994 | 없음 |
| `zerodce` | Zero-DCE | Guo et al., CVPR 2020 | 공식 저장소 |
| `zerodcepp` | Zero-DCE++ | Li et al., TPAMI 2021 | `weights/Epoch99.pth` (보유) |
| `sci` | SCI | Ma et al., CVPR 2022 | 공식 저장소 (easy/medium/difficult 중 선택) |
| `ruas` | RUAS | Liu et al., CVPR 2021 | 공식 저장소 (lol/upe/dark 중 선택) |
| `nightenh` | Night-Enhancement | Jin et al., ECCV 2022 | 공식 저장소 (LOL용 / 빛 번짐 억제용) |

- Gamma·CLAHE는 대표값 하나씩만 본 실험에 넣는다 (이전 실험에서 약하게 처리할수록 결과가 좋았음 → Gamma 1.5, CLAHE clip 2.0). 스윕은 설정으로 켤 수 있게 둔다.
- 가중치가 여러 개인 모델(SCI, RUAS, Night-Enhancement)은 **야간 도로에 가장 가까운 것 하나**를 본 실험에 쓰고, 나머지는 설정으로 바꿀 수 있게 둔다. 선택 이유는 결과 문서에 적는다.
- 모든 모델은 공개된 사전학습 가중치를 그대로 쓴다.

### 2.2 탐지 모델 (측정 도구, 고정)

- `dronefreak/bdd100k-yolo26s`, `dronefreak/bdd100k-yolo26m` (BDD100K 공식 train으로 학습, 학습 이미지 크기 960, AGPL-3.0)
- 입력 크기 640, 960 두 가지
- 재학습하지 않는다

### 2.3 데이터셋

| 이름 | 역할 | 정답 | 평가 클래스 | 상태 |
|---|---|---|---|---|
| BDD100K 야간 | 메인 (블랙박스) | 2D 박스 | person, bicycle, car, motorcycle, bus, truck | 보유. **공식 val로 확인됨** (HF 사본 README의 이름이 `bdd100k-validation`, 10,000장) |
| DarkDriving | 향상성 측정 (낮·밤 정렬) | 2D 박스(Car) + 정렬된 낮 이미지 | car만 | 받아야 함 (테스트 3,632쌍) |
| ACDC 야간 | 실제 도로 야간 | 파노픽 → 박스로 변환 | person, bicycle, car, motorcycle, bus, truck | 받아야 함 (사용자 회원가입 필요할 수 있음) |
| AI Hub 고정밀 야간 도심도로 | 국내 도로 | 폴리곤 → 박스로 변환 | person, bicycle, car, motorcycle, bus, truck | 신청·승인 필요 (사용자) |
| ExDark | 보조 | 2D 박스 | person, bicycle, car, motorcycle, bus (truck 없음) | 보유 |

- 데이터셋마다 **N장(기본 300)을 시드 고정으로 뽑는다.** 모든 조건이 같은 이미지 목록을 쓴다.
- 데이터가 없는 데이터셋은 실험에서 건너뛰고 기록한다 (AI Hub 승인 지연 대비).

---

## 3. 핵심 설계 결정

### 3.1 외부 개선 모델을 가져오는 방법 (확정: B)

| 방법 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **A. 코드 복사** | 각 저장소의 네트워크 정의만 `src/enhance/`에 옮겨 적음 (Zero-DCE++와 같은 방식) | 단순, 테스트 쉬움 | RUAS 등은 **라이선스 표기가 없어서**, 그 코드를 우리 GitHub 저장소에 올리면 문제가 될 수 있음 |
| **B. 원본 저장소를 받아서 가져다 씀 (선택)** | `scripts/fetch_third_party.py`가 공식 저장소를 `third_party/<이름>/`에 받음(git 제외). 우리 쪽 래퍼가 그 안의 모델 클래스만 import해서 같은 인터페이스로 감쌈 | 원본 코드를 고치지 않음, 라이선스 문제 없음(우리 저장소에 안 올라감), 하나의 Python 환경 | 오래된 코드(RUAS: PyTorch 0.4)는 import할 때 작은 호환 처리가 필요할 수 있음 |
| C. 원본 테스트 스크립트를 따로 실행 | 저장소마다 자기 환경에서 `test.py` 실행 | 원본과 가장 같음 | 환경이 여러 개, 시간 측정 조건이 달라져 **공정한 속도 비교가 안 됨** |

선택: **B** (사용자 결정). 공정한 시간 측정(같은 프로세스, 같은 GPU 동기화)이 가능하고, 라이선스가 불분명한 코드를 우리 저장소에 넣지 않는다.
Zero-DCE++는 이미 A 방식으로 구현되어 있고 원 저장소가 비상업 연구 용도를 허용하므로 그대로 둔다.

### 3.2 같은 해상도로 맞추기

- 모든 이미지를 **긴 변 1280px 이하로 줄인 뒤** 개선·탐지·화질 평가를 한다. 정답 박스도 같은 비율로 줄인다.
  - BDD 1280×720(그대로), ACDC 1920×1080 → 1280×720, AI Hub 1920×1200 → 1280×800, DarkDriving 2448×2048 → 1280×1071, ExDark는 크기 제각각
- 이유: (1) 모델마다 같은 크기로 속도를 비교, (2) ExDark·DarkDriving 큰 이미지의 GPU 메모리 부족 방지, (3) "모든 조건 같은 이미지" 원칙 유지
- 설정 `preprocess.max_side`로 바꿀 수 있게 둔다.
- 개선 모델이 입력 크기 제약(배수 등)이 있으면 래퍼 안에서 패딩하고 결과를 원래 크기로 잘라낸다 (Zero-DCE++와 같은 방식).

### 3.3 탐지 클래스 맞추기

- 공통 클래스는 지금과 같은 6개: person, bicycle, car, motorcycle, bus, truck
- BDD 모델 출력 → 공통: person·rider → person, bike → bicycle, car, motor → motorcycle, bus, truck. train·traffic light·traffic sign은 버림
  - 정확한 클래스 이름은 가중치 파일의 `names`에서 읽어 매핑하고, 매핑에 없는 이름이 있으면 시작할 때 오류를 낸다
- 데이터셋 정답 → 공통: 데이터셋별 매핑표(`src/data/classes.py`)에 추가
- 데이터셋에 없는 클래스는 집계에서 뺀다 (DarkDriving은 car만, ExDark는 truck 없음)

### 3.4 시간 측정 분리

지난 실험의 Zero-DCE++ 49ms는 논문 수치(1080p 1.6ms)보다 약 30배 길었다. 측정 구간에 변환·전송이 섞였을 가능성이 높으므로 나눈다.

- 개선기 인터페이스를 세 단계로 나눈다: `preprocess`(numpy → 텐서, GPU 전송) / `infer`(모델 순전파) / `postprocess`(텐서 → numpy)
- 각 단계 전후에 CUDA 동기화를 하고 따로 기록한다. 전통 기법은 `infer`만 있다.
- 탐지도 Ultralytics가 주는 preprocess / inference / postprocess 시간을 같이 기록한다.
- 워밍업 10회, 평균·중앙값·p95를 남긴다. GPU 메모리 최대치(`torch.cuda.max_memory_allocated`)도 기록한다.

---

## 4. 시스템 구조

```
CCTV/
├─ configs/default.yaml           # 데이터셋, 조건, 탐지 모델·입력 크기, 평가 설정
├─ src/
│  ├─ data/
│  │  ├─ base.py                  # (유지) Sample, select_samples
│  │  ├─ classes.py               # (수정) 데이터셋별·탐지 모델별 매핑 추가
│  │  ├─ bdd100k.py, exdark.py    # (유지)
│  │  ├─ darkdriving.py           # (신규) 밤 이미지 + 박스 + 정렬된 낮 이미지 경로
│  │  ├─ acdc.py                  # (신규) 야간, 파노픽 → 인스턴스 박스
│  │  ├─ aihub_night.py           # (신규) 폴리곤 → 박스
│  │  └─ resize.py                # (신규) 긴 변 제한 + 박스 스케일
│  ├─ enhance/
│  │  ├─ base.py                  # (수정) preprocess / infer / postprocess 3단계
│  │  ├─ gamma.py, clahe.py       # (유지, 3단계 인터페이스에 맞춤)
│  │  ├─ zerodce.py               # (수정) Zero-DCE++ 유지 + 원본 Zero-DCE 추가
│  │  ├─ sci.py, ruas.py, nightenh.py   # (신규) third_party 래퍼
│  │  ├─ realesrgan.py            # (삭제)
│  │  └─ registry.py              # (수정)
│  ├─ detect/yolo.py              # (수정) BDD 모델 클래스 매핑, 단계별 시간
│  ├─ eval/
│  │  ├─ matching.py              # (유지)
│  │  ├─ metrics.py               # (수정) mAP@50-95, 객체 크기별(소·중·대)
│  │  ├─ iqa.py                   # (신규) pyiqa로 7개 지표
│  │  └─ analysis.py              # (신규) Δ 계산, 지표-ΔmAP 상관, 빠른/확실한 모델 선택
│  ├─ frames.py                   # (신규) 영상 → 지정 프레임 이미지
│  ├─ visualize.py                # (신규) 비교 이미지(원본·개선본·탐지 박스 격자)
│  ├─ timing.py, io_utils.py, imageio.py, config.py   # (유지, timing은 단계별로 확장)
├─ scripts/
│  ├─ fetch_third_party.py        # (신규) 공식 저장소 받기 (커밋 고정)
│  ├─ download_*.py               # (유지 + DarkDriving·ACDC 안내 추가)
│  ├─ run_experiment.py           # (수정) 개선 1회 → 탐지 4회(2모델×2크기) + 화질 지표
│  ├─ analyze.py                  # (신규) 결과 폴더 → 요약표·상관표·그래프
│  └─ compare_frame.py            # (신규) 영상/이미지 → 비교 이미지
├─ app/streamlit_app.py           # (신규) 결과 뷰어 + 비교 이미지 데모
├─ third_party/                   # (git 제외) 외부 저장소
├─ weights/                       # (git 제외)
└─ results/<run_id>/
```

## 5. 데이터 흐름 (`run_experiment.py`)

```
설정 로드 → 데이터셋별 표본 추출(시드 고정) → 긴 변 1280으로 맞춤
for 데이터셋:
  for 개선 조건 (8개):
    워밍업
    for 이미지:
      개선 (preprocess / infer / postprocess 시간 따로)
      화질 지표 7개 (NR은 전부, FR은 DarkDriving만 — 정렬된 낮 이미지와 비교)
      for 탐지 모델 (26s, 26m) × 입력 크기 (640, 960):
        탐지 (단계별 시간) → 매칭 → 이미지별 TP/FP/FN, 점수 기록
      대표 이미지 몇 장은 비교 이미지 저장
    집계
저장: results/<run_id>/
  detection_metrics.csv   # dataset, condition, detector, imgsz, class, size_bucket, P, R, F1, AP50, AP50_95, TP, FP, FN
  iqa_metrics.csv         # dataset, condition, psnr, ssim, lpips, niqe, brisque, musiq, clipiqa (평균·중앙값)
  timing.csv              # dataset, condition, 단계별 ms (평균·중앙값·p95), gpu_mem_peak
  per_image.jsonl         # 이미지별 상세
  run_meta.json           # 설정 전체, 시드, 패키지 버전, GPU, 외부 저장소 커밋, 제외 이미지 수, 건너뛴 데이터셋
  samples/                # 비교 이미지
```

- 개선된 이미지는 한 번만 만들고 탐지 4번(2모델 × 2크기)과 화질 지표에 같이 쓴다.
- `analyze.py`가 결과 폴더를 읽어 원본 대비 변화량, 상관관계, 빠른/확실한 모델을 계산한다.

---

## 6. 평가 지표

### 6.1 처리 성능
- 개선: preprocess / infer / postprocess / 합계 (ms, 평균·중앙값·p95), GPU 메모리 최대치
- 탐지: preprocess / inference / postprocess (ms)
- 개선 + 탐지 합계 → FPS
- 참고 열: 논문에 나온 파라미터 수·FLOPs (출처 표기)
- 실시간 기준: 프레임당 33.3ms (30fps)

### 6.2 정확성 (탐지)
- P/R/F1 (conf 0.25, IoU 0.5), AP50, AP50-95 (IoU 0.50~0.95, 0.05 간격, 클래스 평균)
- 원본 대비 변화량 ΔP, ΔR, ΔF1, ΔmAP50, ΔmAP50-95
- 나눠 보기: 데이터셋별, 클래스별, 객체 크기별 (COCO 기준: 소 < 32², 중 32²~96², 대 > 96², 1280 기준 해상도에서)
- 해석 주의: BDD는 탐지 모델 학습과 같은 도메인, 나머지 4개는 처음 보는 도메인 → 따로 보고

### 6.3 향상성 (화질, pyiqa)
| 지표 | 종류 | 좋은 방향 | 적용 |
|---|---|---|---|
| PSNR | 정답 기반 | ↑ | DarkDriving (낮 이미지와 비교) |
| SSIM | 정답 기반 | ↑ | DarkDriving |
| LPIPS | 정답 기반 | ↓ | DarkDriving |
| NIQE | 정답 없음 | ↓ | 5개 전부 |
| BRISQUE | 정답 없음 | ↓ | 5개 전부 |
| MUSIQ | 정답 없음 | ↑ | 5개 전부 |
| CLIP-IQA | 정답 없음 | ↑ | 5개 전부 |

- 지표 신뢰성 확인: 데이터셋마다 "개선 조건별 화질 지표 평균"과 "ΔmAP50" 사이의 스피어만 순위 상관계수를 계산해 표로 낸다.

---

## 7. 빠른 모델 vs 확실한 모델 선택 규칙

- 그래프: x = 개선 + 탐지 합계 시간(ms), y = ΔmAP50 (데이터셋별 + 전체 평균)
- **빠른 모델:** 합계 시간 ≤ 33.3ms인 조건 중 ΔmAP50 평균이 가장 큰 것
- **확실한 모델:** 시간과 상관없이, 데이터셋별 ΔmAP50 순위의 평균이 가장 좋은 것 (한 데이터셋에서만 좋은 모델이 뽑히지 않도록 순위 평균 사용)
- 기준 탐지 설정은 YOLO26s·640으로 하고, 나머지 3개 설정에서도 같은 결과가 나오는지 표로 보여준다.

---

## 8. 오류 처리

| 상황 | 동작 |
|---|---|
| 데이터셋 경로 없음 | 그 데이터셋만 건너뛰고 `run_meta.json`에 기록 (전체 중단하지 않음) |
| 외부 저장소·가중치 없음 | 그 개선 조건만 건너뛰고 기록 |
| 개선 중 GPU 메모리 부족 | 타일 처리로 다시 시도 → 실패하면 그 이미지는 **모든 조건에서** 제외해 이미지 목록을 같게 유지하고 기록 |
| 라벨 파싱·이미지 읽기 실패 | 그 이미지 제외, 기록 |
| 탐지 모델 클래스 이름이 매핑에 없음 | 시작할 때 중단하고 이름 목록을 보여줌 |
| CUDA 없음 | CPU로 실행하고 시간 결과에 장치 정보를 남김 |

---

## 9. 테스트

| 종류 | 대상 |
|---|---|
| 단위 | 새 로더 3개 (작은 가짜 라벨 파일로 박스·클래스 변환 확인), 리사이즈 + 박스 스케일 |
| 단위 | mAP50-95, 객체 크기별 집계 (손으로 만든 박스로 값 검증) |
| 단위 | 클래스 매핑 (BDD 모델 이름 → 공통, 모르는 이름이면 오류) |
| 단위 | 개선기 3단계 인터페이스: 출력 크기·dtype 유지, 시간 키 존재 |
| 단위 | 분석: Δ 계산, 스피어만 상관, 빠른/확실한 모델 선택 규칙 |
| 통합 (가중치 있을 때만) | 외부 개선 모델마다 공식 가중치가 오류 없이 로드되고 한 장을 처리 |
| 통합 (pyiqa 있을 때만) | 같은 이미지 두 장 → PSNR 무한대/SSIM 1/LPIPS 0 |
| 스모크 | 가짜 탐지기·가짜 데이터로 전체 흐름과 결과 파일 생성 |

- PyTorch가 필요 없는 테스트는 Claude 작업 환경에서도 돌리고, GPU가 필요한 것은 사용자 PC에서 돌린다.

---

## 10. 구현 페이즈 (초안)

| 페이즈 | 내용 |
|---|---|
| P1 정리 | Real-ESRGAN 제거, 설정 구조 개편, 리사이즈, 클래스 매핑 확장, 시간 측정 3단계 |
| P2 탐지 | BDD YOLO26s·m 로드, 클래스 매핑, mAP50-95·크기별 지표 |
| P3 개선 모델 | third_party 받기 스크립트, Zero-DCE, SCI, RUAS, Night-Enhancement 래퍼 + 가중치 로드 테스트 |
| P4 데이터셋 | DarkDriving, ACDC, AI Hub 로더 (데이터가 생기는 순서대로) |
| P5 화질 지표 | pyiqa 7개 지표, DarkDriving 정답 기반 비교 |
| P6 분석 | Δ, 상관관계, 빠른/확실한 모델 선택, 그래프 |
| P7 실험 | 사용자 PC에서 전체 실험 |
| P8 결과물 | 영상 → 프레임 비교 이미지, Streamlit 화면, 결과 문서 |

각 페이즈가 끝날 때마다 `work_history.md`에 이어서 기록한다.

---

## 11. 외부 의존성과 라이선스

- 탐지: Ultralytics YOLO26 / BDD100K 학습 가중치 — AGPL-3.0. BDD100K 데이터는 교육·연구·비영리 용도
- 개선: Zero-DCE·Zero-DCE++ (비상업 연구), SCI (저장소 확인 필요), RUAS (라이선스 표기 없음 → 우리 저장소에 코드 넣지 않음), Night-Enhancement (학술·비상업 MIT)
- 데이터: DarkDriving (학술·비상업), ACDC (사이트 조건 확인 필요), AI Hub (AI Hub 이용정책), ExDark (BSD-3-Clause)
- 화질 지표: pyiqa
- 수업 과제 용도로는 모두 사용 가능. 외부 공개 배포 시 각 조건을 다시 확인한다.
