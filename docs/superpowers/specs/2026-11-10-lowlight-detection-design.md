# 저조도 도로 영상 화질 개선 × 객체 탐지 비교 시스템 — 설계 스펙

- 작성일: 2026-11-10
- 프로젝트: 컴퓨터비전 팀프로젝트
- 근거 문서: 팀프로젝트 제안서 (1)~(5)절
- 상태: 설계 승인됨 (구조 A) → 스펙 검토 대기

---

## 1. 목표

야간·저조도 도로 영상에 여러 화질 개선 기법을 적용한 뒤, **동일한 탐지 모델과 동일한 평가 조건**에서
탐지 정확도(Precision / Recall / F1, mAP@0.5)와 처리 속도(화질 개선 시간, 탐지 시간을 분리)를 비교한다.
이를 통해 "화질 개선 효과 ↔ 탐지 정확도 ↔ 연산 비용"의 관계를 분석한다.

### 성공 기준
1. 2개 데이터셋 × 5개 조건(원본 + 4개 기법)에 대한 지표표가 CSV/JSON으로 생성된다.
2. 지표는 데이터셋별 · 조건별 · 클래스별로 분리되어 나온다.
3. 기법별 화질 개선 시간과 탐지 시간이 따로 기록된다 (평균, 중앙값, ms/프레임).
4. 웹 화면에서 원본과 개선본, 탐지 결과를 나란히 비교하고 지표를 표·그래프로 볼 수 있다.
5. 라벨 없는 영상(시연용)에 대해 비교 영상을 내보낼 수 있다.

### 범위 밖 (이번 스펙에서 하지 않음)
- AI Hub 야간 도심도로 데이터 (신청 전). 단, 로더를 파일 하나로 추가할 수 있는 구조는 유지한다.
- 새로운 화질 개선 모델 설계·학습.
- 실시간 스트리밍 CCTV 연동.

---

## 2. 확정된 결정 사항

| 항목 | 결정 | 이유 |
|---|---|---|
| 전체 구조 | **A. 배치 실험 + 결과 뷰어 분리** | 실험 재현이 쉽고, 발표 중 웹이 멈출 위험이 적음 |
| 탐지 모델 | Ultralytics **YOLO26** 사전학습 모델 (기본 `yolo26s`, 설정으로 변경 가능) | 제안서 (3)절. 2026-01-14 정식 공개, 학습·추론·export 지원 |
| 화질 개선 기법 | **Gamma, CLAHE, Zero-DCE++, Real-ESRGAN** | 제안서 (2)②의 3개 + (3)절의 Real-ESRGAN을 4번째 비교 기법으로 편입 |
| Real-ESRGAN 처리 | 업스케일 후 **원래 해상도로 복원**해서 탐지기에 입력 | 탐지기 입력 크기를 모든 조건에서 같게 유지 |
| YOLO26 파인튜닝 | **선택 과제** (시간이 남을 때) — BDD100K 야간 train 사용 | 제안서의 "Real-ESRGAN을 YOLO26에 파인튜닝"은 성립하지 않으므로 분리 |
| 평가 데이터셋 | **BDD100K 야간** (메인), **ExDark** (일반화 확인) | 둘 다 박스 라벨 있음, 즉시 확보 가능 |
| 결과 집계 | 데이터셋을 **합치지 않고 따로** 보고 | 도메인이 달라서 합치면 기법별 차이가 묻힘 |
| 실행 환경 | 로컬 Windows PC, **NVIDIA RTX 3070 (8GB)**, VS Code. Colab은 백업 | 사용자 확인 |
| 웹 UI | **Streamlit** | 표·그래프·이미지 비교를 적은 코드로 구현 가능 |
| 작업 폴더 | 로컬 `CCTV` 프로젝트 폴더 (Windows) | 사용자 지정 |

---

## 3. 시스템 구조

```
CCTV/
├─ configs/
│   └─ default.yaml          # 실험 조건 (데이터셋, 표본 수, 시드, 기법·파라미터, conf, IoU)
├─ src/
│   ├─ data/
│   │   ├─ base.py           # Sample(image_path, boxes[N,4] xyxy, labels[N]), Dataset 인터페이스
│   │   ├─ classes.py        # 공통 클래스 정의 + 데이터셋별 매핑표
│   │   ├─ bdd100k.py        # BDD100K det JSON 로더 (timeofday == night 필터)
│   │   └─ exdark.py         # ExDark bbGt 텍스트 로더
│   ├─ enhance/
│   │   ├─ base.py           # Enhancer 인터페이스: name, params, enhance(img_bgr)->img_bgr
│   │   ├─ gamma.py
│   │   ├─ clahe.py
│   │   ├─ zerodce.py        # Zero-DCE++ (PyTorch, 가중치 파일 필요)
│   │   ├─ realesrgan.py     # Real-ESRGAN (PyTorch, 가중치 파일 필요, 타일 처리)
│   │   └─ registry.py       # 이름 → Enhancer 생성
│   ├─ detect/
│   │   └─ yolo.py           # YOLO26 래퍼: detect(img)->(boxes, labels, scores), COCO→공통 클래스
│   ├─ eval/
│   │   ├─ matching.py       # 클래스 일치 + IoU≥0.5 그리디 매칭 (점수 내림차순)
│   │   └─ metrics.py        # P/R/F1, AP@0.5, 집계
│   ├─ timing.py             # 워밍업, CUDA 동기화, 구간별 측정
│   ├─ video.py              # 프레임 추출(간격·해상도), 비교 영상 내보내기
│   └─ io_utils.py           # CSV/JSON 저장, 실행 메타데이터 기록
├─ scripts/
│   ├─ run_experiment.py     # 배치 실험 실행 → results/<run_id>/
│   ├─ export_video.py       # 라벨 없는 영상 → 원본|개선본 비교 영상
│   └─ download_weights.md   # 가중치 받는 방법 안내
├─ app/
│   └─ streamlit_app.py      # 결과 뷰어 + 단일 이미지/영상 데모
├─ tests/                    # 단위 테스트 + 스모크 테스트 + 작은 라벨 샘플
├─ weights/                  # (git 제외) 모델 가중치
├─ datasets/                 # (git 제외) BDD100K, ExDark 원본
├─ results/                  # 실험 결과
└─ requirements.txt
```

### 각 단위의 책임
- **data**: 원본 데이터 형식을 공통 `Sample`로 바꾼다. 클래스 매핑은 `classes.py` 한 곳에서만 관리한다.
- **enhance**: 입력과 같은 크기·같은 형식(BGR uint8)의 이미지를 돌려준다. 내부에서 GPU를 쓰더라도 인터페이스는 같다.
- **detect**: 탐지 결과를 공통 클래스 체계로 돌려준다. 공통 클래스에 없는 COCO 클래스는 버린다.
- **eval**: 탐지 결과와 정답만 보고 지표를 계산한다. 모델이나 이미지에 의존하지 않는다.
- **scripts/run_experiment.py**: 위 단위들을 조합할 뿐, 계산 로직은 갖지 않는다.

---

## 4. 데이터 흐름 (run_experiment.py)

```
config 로드 → 데이터셋별 표본 추출(시드 고정)
  for 데이터셋:
    for 조건 in [원본, gamma(p...), clahe(p...), zerodce, realesrgan]:
      enhancer 워밍업, detector 워밍업
      for 이미지:
        img = 읽기
        t0 → out = enhancer(img) → t1        # 화질 개선 시간
        t1 → dets = detector(out) → t2       # 탐지 시간
        매칭(dets, GT) → 이미지별 TP/FP/FN 기록
        (옵션) 대표 이미지 N장은 개선본 + 탐지 박스 그림 저장
      조건별 집계 → 지표 행 추가
  저장: results/<run_id>/
        metrics.csv        # dataset, condition, params, class, P, R, F1, AP50, TP, FP, FN
        timing.csv         # dataset, condition, enhance_ms_mean/median, detect_ms_mean/median
        per_image.jsonl    # 이미지별 상세 결과
        run_meta.json      # config 전체, 시드, 패키지 버전, GPU 이름, 제외된 이미지 수
        samples/           # 대표 비교 이미지
```

---

## 5. 실험 조건 (공정 비교 원칙)

### 5.1 공통 클래스와 매핑
공통 클래스: `person, bicycle, car, motorcycle, bus, truck` (YOLO의 COCO 클래스와 그대로 대응)

| 공통 클래스 | BDD100K | ExDark |
|---|---|---|
| person | pedestrian, rider | People |
| bicycle | bicycle | Bicycle |
| car | car | Car |
| motorcycle | motorcycle | Motorbike |
| bus | bus | Bus |
| truck | truck | (없음 → 이 데이터셋에서는 평가 제외) |

- BDD100K의 train, traffic light, traffic sign은 제외한다.
- ExDark는 위 5개 클래스 중 하나 이상이 있는 이미지만 표본 대상으로 삼는다.
- 데이터셋에 해당 클래스가 없으면, 그 클래스의 탐지 결과를 FP로 세지 않고 집계에서 빼는 방식으로 처리한다.

### 5.2 표본
- 데이터셋마다 **300장**, 시드 고정 (설정으로 변경 가능).
- BDD100K: 공식 val 세트에서 `timeofday == night`만 사용.
- 모든 조건이 **같은 이미지 목록**을 쓴다.

### 5.3 기법과 파라미터 (초기값, 설정으로 변경 가능)
| 조건 | 구현 | 파라미터 |
|---|---|---|
| 원본 | 그대로 | — |
| Gamma | LUT로 `out = 255 · (in/255)^(1/γ)` | γ ∈ {1.5, 2.0, 2.5} |
| CLAHE | LAB 변환 후 L 채널에만 적용 | clipLimit ∈ {2.0, 3.0, 4.0}, tileGrid 8×8 |
| Zero-DCE++ | 공식 PyTorch 구현 + 사전학습 가중치 | scale_factor 1 |
| Real-ESRGAN | 공식 구현 + 사전학습 가중치 | 모델 RealESRGAN_x2plus, 타일 처리, 출력은 원래 해상도로 리사이즈 |

### 5.4 탐지·평가
- 탐지 모델, 입력 크기(imgsz=640), 신뢰도 임계값(conf=0.25)은 **모든 조건에서 고정**.
- 매칭: 같은 클래스이고 IoU ≥ 0.5이면 TP. 점수가 높은 탐지부터 그리디하게 매칭하고, 하나의 정답은 한 번만 매칭된다.
- P/R/F1은 conf=0.25 기준, AP@0.5는 전체 점수 범위로 계산한다.
- 클래스별 지표와 전체(micro) 지표를 둘 다 낸다.

### 5.5 시간 측정
- 각 조건 시작 전에 워밍업 5회.
- GPU를 쓰는 구간은 측정 전후로 `torch.cuda.synchronize()`.
- 이미지 읽기·저장 시간은 측정에서 뺀다.
- 평균과 중앙값을 모두 기록한다.

---

## 6. 웹 화면 (Streamlit)

1. **결과 비교 탭**: `results/` 안의 실행(run)을 고른다 → 데이터셋·조건별 지표표, F1·Recall 막대그래프, "정확도 vs 처리 시간" 산점도.
2. **이미지 비교 탭**: 표본 이미지를 골라 원본과 각 조건의 개선본을 나란히 보고, 탐지 박스와 정답 박스를 겹쳐서 본다.
3. **데모 탭**: 이미지 또는 영상을 업로드 → 기법과 파라미터를 고르면 즉시 처리해서 전후 비교와 처리 시간을 보여 준다. 영상은 프레임 간격과 해상도를 지정할 수 있다.

결과 뷰어는 저장된 파일만 읽는다. 무거운 실험은 웹에서 돌리지 않는다.

---

## 7. 오류 처리

| 상황 | 동작 |
|---|---|
| 가중치 파일 없음 (Zero-DCE++, Real-ESRGAN, YOLO26) | 해당 조건을 건너뛰고 이유를 로그와 `run_meta.json`에 기록. 나머지 실험은 계속 |
| 라벨 파싱 실패 / 이미지 읽기 실패 | 그 이미지를 제외하고 로그를 남김. 제외 수를 `run_meta.json`에 기록 |
| Real-ESRGAN GPU 메모리 부족 | 타일 크기를 절반으로 줄여 재시도 (최소 128). 그래도 실패하면 그 이미지를 제외로 기록 |
| CUDA를 쓸 수 없음 | CPU로 실행하고 경고를 출력. 시간 결과에 장치 정보를 함께 기록 |
| 데이터셋 경로가 없음 | 시작 시점에 즉시 중단하고 필요한 경로를 안내 |

---

## 8. 테스트

| 종류 | 대상 | 비고 |
|---|---|---|
| 단위 | `eval/matching.py`, `eval/metrics.py` | 손으로 만든 박스로 TP/FP/FN, P/R/F1, AP 값을 직접 검증 |
| 단위 | `data/bdd100k.py`, `data/exdark.py` | `tests/fixtures/`의 작은 라벨 파일로 파싱·클래스 매핑 검증 |
| 단위 | `enhance/gamma.py`, `enhance/clahe.py` | 출력 크기·dtype 유지, 어두운 이미지의 평균 밝기 증가 |
| 단위 | `enhance/zerodce.py`, `enhance/realesrgan.py` | 출력 크기·dtype 유지 (가중치 없으면 skip) |
| 스모크 | `run_experiment.py` | 가짜 탐지기·가짜 데이터 3장으로 전체 흐름과 결과 파일 생성 확인 |

- PyTorch가 필요 없는 테스트는 Claude 작업 환경에서도 돌린다.
- GPU가 필요한 테스트와 실제 실험은 사용자 PC(RTX 3070)에서 돌린다.

---

## 9. 외부 의존성과 라이선스 메모

- BDD100K: 연구·교육용 라이선스. 다운로드 시 동의 필요.
- ExDark: 공식 GitHub에서 배포 (BSD-3-Clause 표기).
- Zero-DCE / Zero-DCE++: 비상업적 연구 목적으로만 사용 가능. 수업 프로젝트 용도로는 문제없음.
- Real-ESRGAN, Ultralytics YOLO26: 각 저장소 라이선스 따름. Ultralytics는 AGPL-3.0이므로 결과물을 외부에 배포할 경우 확인 필요.
