# 저조도 화질 개선 × 객체 탐지 비교 시스템 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BDD100K 야간·ExDark 이미지에 Gamma/CLAHE/Zero-DCE++/Real-ESRGAN을 적용하고, 같은 YOLO26 탐지기·같은 평가 조건으로 정확도와 처리 시간을 비교하는 배치 실험 + Streamlit 뷰어를 만든다.

**Architecture:** 구조 A. `src/`의 독립 단위(data / enhance / detect / eval / timing / io)를 `scripts/run_experiment.py`가 조합해 `results/<run_id>/`에 CSV·JSON을 쓴다. 웹 화면은 저장된 결과만 읽고, 단일 이미지·영상 데모만 즉석 처리한다.

**Tech Stack:** Python 3.10+, numpy, OpenCV, PyYAML, pandas, pytest / PyTorch(CUDA), ultralytics(YOLO26), realesrgan(+basicsr) / Streamlit, matplotlib

**Spec:** `docs/superpowers/specs/2026-11-10-lowlight-detection-design.md`

## Global Constraints

- 공통 클래스 순서 고정: `["person", "bicycle", "car", "motorcycle", "bus", "truck"]` (id 0~5)
- COCO → 공통: `{0:0, 1:1, 2:2, 3:3, 5:4, 7:5}`, 그 외 COCO 클래스는 버린다
- 표본: 데이터셋마다 `num_samples: 300`, `seed: 42`, 모든 조건이 같은 이미지 목록 사용
- 탐지기: `yolo26s.pt`, `imgsz=640`. 탐지는 `conf_min=0.001`로 받고, P/R/F1은 `conf_eval=0.25` 이상만 사용, AP@0.5는 전체 점수 범위
- 매칭: 같은 클래스 + IoU ≥ 0.5, 점수 내림차순 그리디, 정답 하나당 한 번
- Enhancer 입출력: BGR uint8, 출력 크기 == 입력 크기
- 시간: 워밍업 5회, CUDA 사용 구간은 측정 전후 동기화, 이미지 읽기·저장 제외, 평균+중앙값(ms)
- 데이터셋 결과는 합치지 않고 데이터셋별로 보고
- 실행 환경: Windows + RTX 3070(8GB). GPU 없는 테스트는 Claude 작업 환경에서도 통과해야 한다
- 작업 기록: 각 Task 완료 시 커밋 대신 프로젝트 루트 `work_history.md`에 **아래로 추가만** 한다 (기존 내용 수정·삭제 금지). 폴더가 git 저장소가 아니므로 git 커밋 단계는 없다

## Review Focus

1. **BDD100K 라벨 버전 차이** — 2018 원본 라벨(`person`, `bike`, `motor`)과 det_20 라벨(`pedestrian`, `bicycle`, `motorcycle`)이 모두 들어올 수 있다. 둘 다 같은 공통 클래스로 매핑되어야 한다 → Task 2에 테스트.
2. **라벨이 하나도 없는 이미지** — BDD 항목에 `labels` 키가 없거나, 표본 이미지에 평가 클래스 박스가 0개인 경우. 크래시 없이 GT 0개로 처리하고, 그 이미지의 탐지는 전부 FP여야 한다 → Task 2, Task 4에 테스트.
3. **ExDark 파일 확장자 대소문자** — `xxx.JPG`, `xxx.jpeg`, `xxx.png` 혼재, 라벨 파일은 `<이미지 파일명>.txt`. 확장자 대소문자가 달라도 짝이 맞아야 한다 → Task 3에 테스트.
4. **데이터셋에 없는 클래스의 탐지** — ExDark에서 YOLO가 truck을 찾은 경우 FP로 세지 않고 집계에서 뺀다 → Task 4에 테스트.
5. **이미지 읽기 실패 / 가중치 없음** — 깨진 이미지는 제외 목록에 기록하고 진행, 가중치 없는 조건은 건너뛰고 `run_meta.json`의 `skipped_conditions`에 이유 기록 → Task 7에 테스트.

---

# Phase 1: 골격 · 데이터 · 지표 · 기본 기법 · 실험 러너

### Task 1: 프로젝트 골격과 설정 로더

**Files:**
- Create: `requirements.txt`, `.gitignore`, `README.md`, `configs/default.yaml`, `src/__init__.py`, `src/config.py`, `pytest.ini`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict` — YAML을 읽어 기본값과 병합한다. 상대 경로는 **프로젝트 루트 기준**으로 해석된 `Path`로 바꾼다. 필수 키 누락 시 `ConfigError(ValueError)`.
- Produces: `PROJECT_ROOT: Path`

- [ ] **Step 1: 실패하는 테스트 작성** — `test_load_default_config`(seed==42, num_samples==300, detector.conf_eval==0.25, conditions 첫 항목 type=="original"), `test_relative_paths_resolved`(datasets.bdd100k.images_dir가 절대 경로), `test_missing_key_raises`(conditions 없는 yaml → ConfigError)
- [ ] **Step 2: 실행해서 실패 확인** — `pytest tests/test_config.py -v` → FAIL (module not found)
- [ ] **Step 3: 구현** — `configs/default.yaml`은 스펙 5장 값 그대로. 키: `seed, num_samples, datasets.{bdd100k,exdark}, detector.{weights,imgsz,conf_eval,conf_min,device}, matching.iou, conditions[], timing.warmup, output.{root,save_samples}`
- [ ] **Step 4: 통과 확인** — `pytest tests/test_config.py -v` → PASS
- [ ] **Step 5: work_history.md에 기록 추가**

### Task 2: 공통 클래스 · Sample · BDD100K 로더

**Files:**
- Create: `src/data/__init__.py`, `src/data/classes.py`, `src/data/base.py`, `src/data/bdd100k.py`
- Test: `tests/test_bdd100k.py`, `tests/fixtures/bdd_mini.json`

**Interfaces:**
- Produces (`classes.py`): `COMMON_CLASSES: list[str]`, `COCO_TO_COMMON: dict[int,int]`, `BDD_TO_COMMON: dict[str,int]`(`pedestrian, person, rider → 0`, `bicycle, bike → 1`, `car → 2`, `motorcycle, motor → 3`, `bus → 4`, `truck → 5`), `EXDARK_TO_COMMON: dict[str,int]`(`People 0, Bicycle 1, Car 2, Motorbike 3, Bus 4`)
- Produces (`base.py`): `@dataclass Sample(image_id: str, image_path: Path, boxes: np.ndarray /*(N,4) float32 xyxy*/, labels: np.ndarray /*(N,) int64*/)`, `class DetectionDataset(ABC)` with `name: str`, `eval_classes: list[int]`, `load() -> list[Sample]`; `select_samples(samples: list[Sample], n: int, seed: int) -> list[Sample]` (image_id 정렬 후 `random.Random(seed).sample`, n ≥ 전체면 전체)
- Produces: `BDD100KDataset(labels_json: Path, images_dir: Path, timeofday: str = "night")`, `name="bdd100k"`, `eval_classes=[0,1,2,3,4,5]`

- [ ] **Step 1: 테스트** — fixture에 4개 항목(night 3, daytime 1; 그중 하나는 `labels` 키 없음; 2018식 `person`/`motor` 라벨 1개; `traffic light` 1개)
  - `test_filters_night_only` → 3개
  - `test_maps_both_label_versions` → `person`과 `pedestrian` 모두 0, `motor` → 3
  - `test_drops_unmapped_categories` → `traffic light` 박스 없음
  - `test_missing_labels_key_gives_empty` → boxes.shape == (0,4)
  - `test_select_samples_deterministic` → 같은 seed면 같은 id 목록, 다른 seed면 다를 수 있음
- [ ] **Step 2: 실패 확인** — `pytest tests/test_bdd100k.py -v`
- [ ] **Step 3: 구현** — `box2d`가 없는 라벨은 무시
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 3: ExDark 로더

**Files:**
- Create: `src/data/exdark.py`
- Test: `tests/test_exdark.py` (픽스처는 `tmp_path`에 생성)

**Interfaces:**
- Produces: `ExDarkDataset(images_dir: Path, annos_dir: Path)`, `name="exdark"`, `eval_classes=[0,1,2,3,4]`. 폴더 구조 `images_dir/<Class>/<img>`, `annos_dir/<Class>/<img>.txt`. 라벨 줄 형식 `Class l t w h ...`(첫 줄 `% bbGt` 주석 무시) → xyxy로 변환
- Produces: `parse_bbgt(text: str) -> list[tuple[str, float, float, float, float]]`

- [ ] **Step 1: 테스트**
  - `test_parse_bbgt` → `"% bbGt version=3\nCar 10 20 30 40 0 0 0 0 0 0 0\n"` → `[("Car",10,20,30,40)]`
  - `test_xywh_to_xyxy` → 박스 `[10,20,40,60]`
  - `test_extension_case_mismatch` → 이미지 `a.JPG`, 라벨 `a.jpg.txt` 여도 짝이 맞음
  - `test_skips_images_without_eval_classes` → `Cup`만 있는 이미지는 결과에 없음
  - `test_missing_anno_file_skipped` → 라벨 없는 이미지 제외, 크래시 없음
- [ ] **Step 2~4: 실패 확인 → 구현(라벨 파일명은 소문자로 비교) → 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 4: 박스 매칭과 지표

**Files:**
- Create: `src/eval/__init__.py`, `src/eval/matching.py`, `src/eval/metrics.py`
- Test: `tests/test_matching.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray` (N,M)
- Produces: `match_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5) -> np.ndarray` — 각 예측의 TP 여부 bool (P,)
- Produces: `@dataclass ImageRecord(image_id: str, pred_labels: np.ndarray, pred_scores: np.ndarray, is_tp: np.ndarray, gt_labels: np.ndarray)`
- Produces: `filter_to_classes(labels: np.ndarray, eval_classes: list[int]) -> np.ndarray` (bool mask)
- Produces: `compute_metrics(records: list[ImageRecord], eval_classes: list[int], conf_eval: float) -> dict[str, dict]` — 키는 클래스 이름들 + `"all"`. 값: `{"precision","recall","f1","ap50","tp","fp","fn","n_gt"}`. `"all"`의 P/R/F1은 micro, `ap50`은 GT가 1개 이상인 클래스 AP의 평균(mAP@0.5). 분모 0이면 0.0
- AP 계산: 클래스별 점수 내림차순 누적 TP/FP → precision envelope(오른쪽에서 최대값 누적) → recall 변화 지점에서 all-point 적분

- [ ] **Step 1: 테스트**
  - `test_iou_identical_is_one`, `test_iou_disjoint_is_zero`, `test_iou_half_overlap` (`[0,0,10,10]` vs `[5,0,15,10]` → 1/3)
  - `test_match_requires_same_class` → 박스 같아도 클래스 다르면 FP
  - `test_one_gt_matched_once` → 같은 GT에 예측 2개면 점수 높은 쪽만 TP
  - `test_greedy_by_score` → 점수 0.9 예측이 먼저 매칭
  - `test_metrics_perfect` → 모든 클래스 P=R=F1=AP=1.0
  - `test_conf_eval_threshold` → 점수 0.1인 TP는 P/R/F1에서 제외(FN 증가), AP에는 반영
  - `test_no_gt_image_all_fp` → GT 0개 이미지의 예측은 전부 FP
  - `test_class_outside_eval_ignored` → eval_classes=[0..4]일 때 label 5 예측은 FP에 포함되지 않음
  - `test_ap_known_value` → 한 클래스, GT 2개, 예측 점수 [0.9 TP, 0.8 FP, 0.7 TP] → AP = 0.5·1.0 + 0.5·(2/3) = 0.8333
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 5: Enhancer 인터페이스 · Gamma · CLAHE · 레지스트리

**Files:**
- Create: `src/enhance/__init__.py`, `src/enhance/base.py`, `src/enhance/gamma.py`, `src/enhance/clahe.py`, `src/enhance/registry.py`
- Test: `tests/test_enhance.py`

**Interfaces:**
- Produces: `class EnhancerUnavailable(RuntimeError)`; `class Enhancer(ABC)`: 속성 `name: str`, `params: dict`, `uses_cuda: bool = False`; `enhance(img: np.ndarray) -> np.ndarray`; `IdentityEnhancer` (name `"original"`)
- Produces: `GammaEnhancer(gamma: float)` name `f"gamma_g{gamma}"`, LUT `255*(i/255)**(1/gamma)` 반올림
- Produces: `ClaheEnhancer(clip_limit: float, tile_grid: int = 8)` name `f"clahe_c{clip_limit}"`, LAB의 L 채널에만 적용
- Produces: `build_enhancers(conditions: list[dict]) -> tuple[list[Enhancer], list[dict]]` — `(사용 가능한 목록, skipped=[{"condition": str, "reason": str}])`. `sweep`의 각 값마다 Enhancer 1개. `zerodce`/`realesrgan`은 Phase 2 전까지 `EnhancerUnavailable("not implemented yet")`로 skipped에 들어간다. 모르는 type은 `ValueError`

- [ ] **Step 1: 테스트**
  - `test_output_shape_dtype` (모든 Enhancer, 입력 (37,53,3) uint8)
  - `test_gamma_brightens_dark_image` → 평균 밝기 증가
  - `test_gamma_one_is_identity`
  - `test_clahe_keeps_shape_and_changes_dark_image`
  - `test_build_enhancers_expands_sweep` → default.yaml 조건 → 이름 `["original","gamma_g1.5","gamma_g2.0","gamma_g2.5","clahe_c2.0","clahe_c3.0","clahe_c4.0"]`, skipped에 zerodce, realesrgan
  - `test_unknown_type_raises`
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 6: 탐지기 래퍼 (YOLO26)

**Files:**
- Create: `src/detect/__init__.py`, `src/detect/base.py`, `src/detect/yolo.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Produces (`base.py`): `@dataclass Detections(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray)`; `class Detector(ABC)`: `uses_cuda: bool`, `detect(img_bgr: np.ndarray) -> Detections`
- Produces: `coco_to_common(boxes, coco_cls, scores) -> Detections` — `COCO_TO_COMMON`에 없는 클래스 제거
- Produces: `YoloDetector(weights: Path, imgsz: int = 640, conf: float = 0.001, device: int | str | None = 0)` — ultralytics는 생성자 안에서 import(없으면 `ImportError` 메시지에 설치 명령 포함), `model.predict(img, imgsz, conf, device, verbose=False)` 결과의 `boxes.xyxy/cls/conf`를 numpy로 → `coco_to_common`. CUDA 없으면 경고 후 `device="cpu"`

- [ ] **Step 1: 테스트** — `test_coco_to_common_maps_and_drops` (cls [0,2,5,7,9,16] → labels [0,2,4,5], 박스·점수 순서 유지), `test_empty_detections` (0개 입력 → shape (0,4)). YoloDetector 자체는 `pytest.importorskip("ultralytics")`로 감싼 `test_yolo_smoke`(사용자 PC에서만 실행)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 7: 시간 측정 · 결과 저장 · 실험 러너

**Files:**
- Create: `src/timing.py`, `src/io_utils.py`, `scripts/__init__.py`, `scripts/run_experiment.py`
- Test: `tests/test_timing.py`, `tests/test_run_experiment.py`

**Interfaces:**
- Consumes: Task 1~6 전부
- Produces: `timed(fn, *args, sync_cuda: bool = False) -> tuple[Any, float]` (결과, ms; `time.perf_counter`; torch 없으면 동기화 생략); `summarize_ms(values: list[float]) -> dict` (`mean`, `median`, `n`)
- Produces: `write_csv(rows: list[dict], path)`, `write_json(obj, path)`, `append_jsonl(obj, path)`, `collect_env() -> dict` (python·numpy·opencv·torch·ultralytics 버전, GPU 이름 또는 "cpu")
- Produces: `run_experiment(cfg: dict, detector: Detector | None = None, datasets: list[DetectionDataset] | None = None, out_root: Path | None = None, run_id: str | None = None) -> Path` — 인자로 받은 detector/datasets가 있으면 그걸 쓰고(테스트용), 없으면 cfg로 생성. 반환은 결과 폴더
- 결과 파일과 열(스펙 4장):
  - `metrics.csv`: `dataset, condition, params, class, precision, recall, f1, ap50, tp, fp, fn, n_gt`
  - `timing.csv`: `dataset, condition, enhance_ms_mean, enhance_ms_median, detect_ms_mean, detect_ms_median, n_images`
  - `per_image.jsonl`: `dataset, condition, image_id, n_pred, n_tp, n_gt, enhance_ms, detect_ms`
  - `run_meta.json`: `config, env, run_id, started_at, finished_at, skipped_conditions, excluded_images[{dataset,image_id,reason}], sample_ids{dataset: [ids]}`
  - `samples/<dataset>/<condition>/<image_id>.jpg`: 처음 `save_samples`장, 예측 박스(초록)와 GT(빨강)를 그린 이미지
- CLI: `python -m scripts.run_experiment --config configs/default.yaml [--num-samples N] [--datasets bdd100k exdark]`
- 루프 순서: 데이터셋 → 조건 → 이미지. 조건 시작마다 첫 이미지로 enhancer·detector 워밍업 `timing.warmup`회. `cv2.imread`가 None이면 excluded에 추가하고 다음 이미지

- [ ] **Step 1: 테스트**
  - `test_timed_returns_result_and_ms`, `test_summarize_ms` ([1,2,3,10] → mean 4.0, median 2.5)
  - `test_smoke_run` — `tmp_path`에 이미지 3장 + 가짜 데이터셋 + GT를 그대로 돌려주는 `OracleDetector` → `metrics.csv`의 `all` 행 f1 == 1.0 (original 조건), 5개 파일 모두 존재, `skipped_conditions`에 zerodce·realesrgan
  - `test_unreadable_image_excluded` — 깨진 파일 1장 포함 → `excluded_images` 길이 1, 나머지 정상 처리
  - `test_missing_dataset_path_raises` — 활성화된 데이터셋 경로가 없으면 실험 시작 전에 `FileNotFoundError`(메시지에 경로 포함). `load_config`는 경로 존재를 검사하지 않는다(데이터 없이 테스트 가능해야 하므로)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: 전체 테스트** — `pytest -q` → 모두 PASS (ultralytics 없는 환경에서는 yolo smoke만 skip)
- [ ] **Step 6: work_history.md 추가 + Phase 1 요약**

**Phase 1 사용자 PC 확인:** `pip install -r requirements.txt` 후 `python -m scripts.run_experiment --config configs/default.yaml --num-samples 20` → original/gamma/clahe 7개 조건 결과 생성

---

# Phase 2: 딥러닝 화질 개선 기법

### Task 8: Zero-DCE++

**Files:** Create `src/enhance/zerodce.py`; Modify `src/enhance/registry.py`; Test `tests/test_zerodce.py`

**Interfaces:**
- Produces: `ZeroDCEEnhancer(weights: Path, scale_factor: int = 1, device: str | None = None)` name `"zerodce"`, `uses_cuda=True`(CUDA 사용 시)
- 네트워크는 공식 `enhance_net_nopool`과 **같은 모듈 이름**(`e_conv1..e_conv7`, 각 `depth_conv`/`point_conv`, number_f=32, 반복 8회 curve)으로 정의해야 공식 `Epoch99.pth`가 그대로 로드된다. 입력은 RGB float [0,1], H·W를 scale_factor 배수로 crop 후 처리하고 원래 크기로 복원
- 가중치 파일이 없거나 torch가 없으면 `EnhancerUnavailable`

- [ ] **Step 1: 테스트** — `importorskip("torch")`; 무작위 초기화 모델로 `test_shape_dtype_preserved`, `test_missing_weights_unavailable`(경로 없음 → EnhancerUnavailable)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC)**
- [ ] **Step 5: work_history.md 추가**

### Task 9: Real-ESRGAN

**Files:** Create `src/enhance/realesrgan.py`; Modify `src/enhance/registry.py`, `requirements.txt`; Test `tests/test_realesrgan.py`

**Interfaces:**
- Produces: `RealESRGANEnhancer(weights: Path, scale: int = 2, tile: int = 400, min_tile: int = 128, half: bool = True)` name `"realesrgan_x2"`
- `realesrgan.RealESRGANer` + `basicsr.archs.rrdbnet_arch.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)`
- **호환 처리:** 최신 torchvision에는 `torchvision.transforms.functional_tensor`가 없어서 basicsr import가 실패한다. import 전에 없으면 `sys.modules`에 `torchvision.transforms.functional`을 그 이름으로 등록
- 출력은 `cv2.resize(..., (w, h), interpolation=cv2.INTER_AREA)`로 입력 크기 복원
- `torch.cuda.OutOfMemoryError` → `torch.cuda.empty_cache()`, tile 절반으로 재시도, `min_tile` 미만이면 예외를 올려 러너가 해당 이미지를 excluded로 기록

- [ ] **Step 1: 테스트** — `importorskip("realesrgan")`; `test_output_same_size_as_input`, `test_missing_weights_unavailable`, `test_oom_halves_tile`(RealESRGANer.enhance를 첫 호출만 OOM 던지는 가짜로 교체 → tile 400→200)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC)**
- [ ] **Step 5: work_history.md 추가**

### Task 10: 가중치 안내와 전체 실험 1회

**Files:** Create `scripts/download_weights.md`; Modify `README.md`

- [ ] **Step 1:** 가중치 3종(yolo26s.pt, Zero-DCE++ Epoch99.pth, RealESRGAN_x2plus.pth)의 공식 출처와 저장 위치(`weights/`) 기록
- [ ] **Step 2:** 사용자 PC에서 `python -m scripts.run_experiment --config configs/default.yaml` (2 × 300장 × 9조건)
- [ ] **Step 3:** `skipped_conditions`가 비어 있고 `metrics.csv`가 2 × 9 × (클래스+all) 행인지 확인
- [ ] **Step 4: work_history.md 추가 + Phase 2 요약**

---

# Phase 3: 비교 영상 · 웹 화면 · (선택) 파인튜닝

### Task 11: 프레임 추출과 비교 영상 내보내기

**Files:** Create `src/video.py`, `scripts/export_video.py`; Test `tests/test_video.py`

**Interfaces:**
- Produces: `iter_frames(path: Path, every_n: int = 1, max_width: int | None = None) -> Iterator[tuple[int, np.ndarray]]` (가로가 max_width보다 크면 비율 유지 축소)
- Produces: `side_by_side(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> np.ndarray`
- Produces: `export_comparison(video_in: Path, video_out: Path, enhancer: Enhancer, detector: Detector | None, every_n: int, max_width: int | None) -> dict` (프레임 수, 평균 enhance/detect ms). 코덱 `mp4v`
- CLI: `python -m scripts.export_video --input X.mp4 --output out.mp4 --condition clahe_c3.0 [--detect] [--every-n 2] [--max-width 1280]`

- [ ] **Step 1: 테스트** — OpenCV로 `tmp_path`에 10프레임 짜리 영상 생성 → `test_iter_frames_every_n`(every_n=3 → 인덱스 [0,3,6,9]), `test_max_width_resize`, `test_side_by_side_shape`(높이 동일, 폭 합), `test_export_creates_file`(Identity enhancer, detector None)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가**

### Task 12: Streamlit 웹 화면

**Files:** Create `app/streamlit_app.py`, `app/views.py`; Test `tests/test_views.py`

**Interfaces:**
- `views.py`(Streamlit 없이 테스트 가능한 순수 함수): `list_runs(results_root: Path) -> list[str]`(최신순), `load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]`, `summary_table(metrics: pd.DataFrame) -> pd.DataFrame`(class=="all" 행만, dataset·condition별 P/R/F1/AP50), `accuracy_vs_time(metrics, timing) -> pd.DataFrame`(dataset, condition, f1, total_ms)
- 탭 3개(스펙 6장): 결과 비교 / 이미지 비교(`samples/`) / 데모(업로드 → 조건 선택 → 즉석 처리, 탐지기는 `st.cache_resource`)
- 실행: `streamlit run app/streamlit_app.py`

- [ ] **Step 1: 테스트** — Task 7 smoke 결과 폴더를 fixture로 만들어 `test_list_runs_sorted`, `test_summary_table_only_all_rows`, `test_accuracy_vs_time_sums_ms`
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: 사용자 PC에서 화면 확인** — 탭 3개가 오류 없이 열림
- [ ] **Step 6: work_history.md 추가**

### Task 13 (선택): YOLO26s 야간 파인튜닝

**Files:** Create `scripts/bdd_to_yolo.py`, `scripts/finetune_yolo.py`; Test `tests/test_bdd_to_yolo.py`

**Interfaces:**
- `bdd_to_yolo(labels_json, images_dir, out_dir, timeofday="night", limit: int | None, seed=42) -> Path`(data.yaml 경로). 라벨은 YOLO txt(정규화 cx cy w h), 클래스는 공통 6개
- `finetune_yolo.py`: `YOLO("weights/yolo26s.pt").train(data=..., imgsz=640, epochs=30, batch=-1)` (batch=-1 → autobatch, 8GB에 맞춤). 학습된 가중치로 `detector.weights`만 바꿔 Task 7 실험을 다시 돌려 비교
- **평가 오염 금지:** 학습은 BDD train split만, 평가 표본은 val split — 겹치지 않음을 `test_no_val_images_in_train` 으로 확인

- [ ] **Step 1~4: 테스트 → 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가 + Phase 3 요약**

---

# Phase 4: 발표용 결과 정리

### Task 14: 그래프 생성

**Files:** Create `scripts/make_figures.py`; Test `tests/test_make_figures.py`

**Interfaces:**
- `make_figures(run_dir: Path, out_dir: Path) -> list[Path]` — `f1_by_condition_<dataset>.png`, `recall_by_class_<dataset>.png`, `f1_vs_time.png`, `summary.csv`. `views.summary_table`, `views.accuracy_vs_time` 재사용
- [ ] **Step 1~4: 테스트(파일 생성 개수) → 실패 확인 → 구현 → 통과 확인**
- [ ] **Step 5: work_history.md 추가 + 최종 요약**
