# 화질 개선 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 야간 차량 카메라 이미지에 화질 개선 모델 7개를 적용하고, 탐지(BDD YOLO26s·m × 640·960)·화질 지표 7개·처리 시간으로 비교해 빠른 모델과 확실한 모델을 뽑는 배치 실험과 결과물(비교 이미지, 웹 화면)을 만든다.

**Architecture:** 기존 배치 실험 구조(src 단위 모듈 + `scripts/run_experiment.py`)를 유지한다. 개선된 이미지는 조건마다 한 번 만들고, 탐지 4회와 화질 지표에 같이 쓴다. 외부 개선 모델은 공식 저장소를 `third_party/`에 받아 모델 클래스만 불러온다(git 제외).

**Tech Stack:** Python 3.11, PyTorch(CUDA), Ultralytics YOLO26, OpenCV, pyiqa, pandas, matplotlib, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-11-12-enhancement-redesign-design.md`

## Global Constraints

- 개선 조건 8개: `original`, `gamma`(γ=1.5), `clahe`(clip 2.0, tile 8), `zerodce`, `zerodcepp`, `sci`, `ruas`, `nightenh`. PairLIE·CLIP-LIT·Real-ESRGAN은 넣지 않는다.
- 탐지 모델: `dronefreak/bdd100k-yolo26s`, `dronefreak/bdd100k-yolo26m`. 입력 크기 640, 960. 재학습 없음. COCO YOLO26s 기준선 없음.
- 공통 클래스 6개: person, bicycle, car, motorcycle, bus, truck (`src/data/classes.py`에서만 관리)
- 모든 이미지는 긴 변 1280px 이하로 줄이고, 정답 박스도 같은 비율로 줄인다 (`preprocess.max_side: 1280`)
- 탐지 평가: conf 0.25(P/R/F1), conf_min 0.001(AP), IoU 0.5, AP50-95는 IoU 0.50~0.95를 0.05 간격(10개)
- 객체 크기: 소 < 32², 중 32²~96², 대 > 96² (줄인 뒤 해상도 기준 넓이)
- 화질 지표 pyiqa 이름: `psnr`, `ssim`, `lpips`(정답 기반, DarkDriving만), `niqe`, `brisque`, `musiq`, `clipiqa`(정답 없음, 전부)
- 시간: 워밍업 10회, 평균·중앙값·p95. 실시간 기준 33.3ms
- 외부 저장소 코드는 우리 저장소에 커밋하지 않는다 (`third_party/`, `weights/`, `datasets/`, `results/`는 git 제외)
- 개인정보(사용자 이름이 들어간 절대 경로 등)를 결과 파일에 남기지 않는다 (`relativize_paths` 사용)
- `requirements.txt`는 ASCII만 쓴다 (한국어 Windows pip가 cp949로 읽음)
- 커밋은 로컬에만 한다. push는 사용자가 한다.
- 각 페이즈가 끝나면 `work_history.md` 맨 아래에 이어서 기록한다 (기존 내용 수정·삭제 금지)

## Review Focus

1. 외부 저장소 여러 개가 모두 `model.py`라는 같은 이름의 모듈을 갖는다 → 나중에 불러온 모델이 앞의 모델을 덮어쓰면 안 된다 (Task 6)
2. 개선 모델이 패딩·다운샘플을 쓰는데 홀수 크기(예: 721×1281) 이미지가 들어온다 → 출력 크기가 입력과 정확히 같아야 박스가 어긋나지 않는다 (Task 3의 계약 테스트, Task 7~10이 사용)
3. 흑백 이미지나 투명 채널이 있는 PNG가 들어온다 (ExDark에 섞여 있음) → 3채널 BGR로 바꿔 처리해야 한다 (Task 2)
4. 특정 클래스나 특정 크기 구간에 정답이 0개다 → AP가 NaN이 되거나 평균을 끌어내리면 안 되고, n_gt=0으로 기록하고 평균에서 빼야 한다 (Task 5)
5. 아주 어둡거나 단색인 원본 이미지에서 NIQE·BRISQUE가 NaN/inf를 낸다 → 그 값은 NaN으로 기록하고, 평균에서 빼되 뺀 개수를 남겨야 한다 (Task 14)

---

# Phase P1: 정리와 공통 기반

### Task 1: 설정 구조 개편, Real-ESRGAN 제거

**Files:**
- Delete: `src/enhance/realesrgan.py`, `tests/test_realesrgan.py`
- Modify: `configs/default.yaml`, `src/config.py`, `src/enhance/registry.py`, `requirements.txt`, `.gitignore`
- Test: `tests/test_config.py`, `tests/test_project_files.py`

**Interfaces:**
- Produces: 설정 키 `preprocess.max_side`(int), `detectors`(list of `{name, weights, hf_repo}`), `detection.imgsz`(list[int]), `detection.conf_eval`, `detection.conf_min`, `detection.device`, `iqa.metrics_nr`(list[str]), `iqa.metrics_fr`(list[str]), `iqa.device`, `timing.warmup`=10, `third_party.root`, `datasets.<name>.enabled`. 기존 `detector` 단일 키는 없어진다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_config.py`에 추가
  - `test_default_config_has_new_sections`: `load_config(default.yaml)` 결과에 `cfg["preprocess"]["max_side"] == 1280`, `[d["name"] for d in cfg["detectors"]] == ["yolo26s_bdd", "yolo26m_bdd"]`, `cfg["detection"]["imgsz"] == [640, 960]`, `cfg["iqa"]["metrics_fr"] == ["psnr", "ssim", "lpips"]`, `cfg["iqa"]["metrics_nr"] == ["niqe", "brisque", "musiq", "clipiqa"]`, `cfg["timing"]["warmup"] == 10`
  - `test_condition_types`: `[c["type"] for c in cfg["conditions"]] == ["original", "gamma", "clahe", "zerodce", "zerodcepp", "sci", "ruas", "nightenh"]`
  - `test_missing_detectors_raises`: `detectors` 키가 없는 YAML → `ConfigError`
  - `tests/test_project_files.py`: `test_requirements_no_realesrgan` (`realesrgan`, `basicsr` 없음, `pyiqa` 있음, ASCII만), `test_gitignore_has_third_party` (`third_party/` 줄 있음)
- [ ] **Step 2: 실행해서 실패 확인** — `pytest tests/test_config.py tests/test_project_files.py -v` → 새 테스트 FAIL
- [ ] **Step 3: 구현**
  - `default.yaml`을 위 키로 다시 쓴다. 조건별 기본값: gamma `gamma: 1.5`, clahe `clip_limit: 2.0, tile_grid: 8`, zerodce `repo: zerodce, weights: third_party/zerodce/Zero-DCE_code/snapshots/Epoch99.pth`, zerodcepp `weights: weights/Epoch99.pth, scale_factor: 1`, sci `repo: sci, variant: difficult`, ruas `repo: ruas, variant: dark`, nightenh `repo: nightenh, variant: delighteffects`
    - 변형 기본값 이유(주석으로 남김): SCI `difficult` = 가장 어두운 장면용, RUAS `dark` = 야간 DarkFace로 학습, Night-Enhancement `delighteffects` = 헤드라이트 등 빛 번짐 억제가 이 모델을 넣은 이유
  - `detectors`: `yolo26s_bdd`(weights `weights/bdd100k-yolo26s.pt`, hf_repo `dronefreak/bdd100k-yolo26s`), `yolo26m_bdd`(weights `weights/bdd100k-yolo26m.pt`, hf_repo `dronefreak/bdd100k-yolo26m`)
  - `src/config.py`: `REQUIRED_KEYS`를 `["seed", "num_samples", "datasets", "detectors", "detection", "conditions"]`로, `DEFAULTS`에 `preprocess.max_side=1280`, `timing.warmup=10`, `third_party.root="third_party"`, `iqa`(위 목록, device 0). `PATH_KEYS`에 `"root"`는 이미 있음 → `third_party.root`도 절대 경로가 되는지 확인
  - `registry.py`에서 realesrgan 빌더 삭제. 아직 없는 `zerodcepp`, `sci`, `ruas`, `nightenh` 빌더는 Task 3·7~10에서 추가하므로, 이 태스크에서는 `BUILDERS`에 없는 type이면 지금처럼 오류 대신 **`skipped`에 "아직 구현되지 않음"으로 넣는다**
  - `requirements.txt`: `realesrgan`, `basicsr` 줄 삭제, `pyiqa` 추가. 첫 줄 주석의 basicsr 문구 삭제
  - `.gitignore`: `third_party/` 추가
- [ ] **Step 4: 통과 확인** — `pytest -q` 전체 → PASS (realesrgan 관련 테스트는 파일째 삭제됨)
- [ ] **Step 5: 커밋** — `git commit -m "refactor: 재설계 설정 구조, Real-ESRGAN 제거"`

### Task 2: 이미지 크기 맞추기와 채널 정리

**Files:**
- Create: `src/data/resize.py`
- Modify: `src/imageio.py`
- Test: `tests/test_resize.py`

**Interfaces:**
- Produces:
  - `resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]` — (줄인 이미지, 배율). 긴 변이 max_side 이하이면 원본 그대로, 배율 1.0. 줄일 때 `cv2.INTER_AREA`
  - `scale_boxes(boxes: np.ndarray, s: float) -> np.ndarray` — xyxy × s, float32
  - `read_image(path) -> np.ndarray | None`가 항상 3채널 BGR uint8을 돌려준다 (흑백 → BGR, BGRA → BGR)

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_resize_long_side`: 2048×2448 → (1071, 1280, 3), 배율 = 1280/2448
  - `test_no_upscale`: 720×1280 → 그대로, 배율 1.0, 같은 객체가 아니라 같은 값
  - `test_scale_boxes`: `[[10,20,110,220]]` × 0.5 → `[[5,10,55,110]]`
  - `test_read_gray_png_returns_bgr`, `test_read_bgra_png_returns_bgr`: tmp에 저장한 흑백·4채널 PNG → shape (h, w, 3), dtype uint8
- [ ] **Step 2: 실패 확인** — `pytest tests/test_resize.py -v` → FAIL
- [ ] **Step 3: 구현** — `read_image`는 `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)` 후 채널 수로 분기
- [ ] **Step 4: 통과 확인** — `pytest tests/test_resize.py tests/test_io_utils.py -v` → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 긴 변 제한 리사이즈, 이미지 채널 정리"`

### Task 3: 개선기 3단계 인터페이스와 단계별 시간 측정

**Files:**
- Modify: `src/enhance/base.py`, `src/enhance/gamma.py`, `src/enhance/clahe.py`, `src/enhance/zerodce.py`, `src/enhance/registry.py`, `src/timing.py`
- Create: `tests/enhancer_contract.py`
- Test: `tests/test_enhance.py`, `tests/test_zerodce.py`, `tests/test_timing.py`

**Interfaces:**
- Produces:
  - `Enhancer` 추상 메서드 3개: `preprocess(img: np.ndarray) -> Any`, `infer(x: Any) -> Any`, `postprocess(y: Any, img_shape: tuple[int, int]) -> np.ndarray`. 공통 메서드 `enhance(img) -> np.ndarray`(세 단계를 이어 부름), `enhance_timed(img) -> tuple[np.ndarray, dict[str, float]]`(키 `pre_ms`, `infer_ms`, `post_ms`, `total_ms`; `uses_cuda`이면 단계마다 CUDA 동기화)
  - 전통 기법(gamma, clahe, original): preprocess/postprocess는 그대로 넘기고 계산은 infer에서
  - `ZeroDCEPPEnhancer` (기존 `ZeroDCEEnhancer`를 이 이름으로 바꿈, 조건 `zerodcepp`)
  - `tests/enhancer_contract.py`: `assert_enhancer_contract(enh: Enhancer) -> None` — 721×1281 랜덤 uint8 BGR 입력에 대해 출력 shape·dtype이 입력과 같고 `enhance_timed`의 네 키가 0 이상인지 확인. Task 7~10이 사용
  - `src/timing.py`: `summarize_ms`가 `p95`도 돌려준다; `gpu_peak_mb() -> float | None`, `reset_gpu_peak() -> None` (torch·CUDA 없으면 None / 아무것도 안 함)

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_enhance.py`: `assert_enhancer_contract(GammaEnhancer(1.5))`, `ClaheEnhancer(2.0, 8)`, `IdentityEnhancer()`; `test_enhance_equals_staged`: `enh.enhance(img)`와 `enh.postprocess(enh.infer(enh.preprocess(img)), img.shape[:2])`가 같은 배열
  - `test_zerodce.py`: 기존 테스트의 클래스 이름을 `ZeroDCEPPEnhancer`로 바꾸고, 가중치가 있을 때만 `assert_enhancer_contract`
  - `test_timing.py`: `summarize_ms([1,2,3,4,100])["p95"]`가 `numpy.percentile(..., 95)`와 같음, 빈 목록이면 0.0
- [ ] **Step 2: 실패 확인** — `pytest tests/test_enhance.py tests/test_zerodce.py tests/test_timing.py -v` → FAIL
- [ ] **Step 3: 구현** — Zero-DCE++의 preprocess = 패딩 + RGB 변환 + 텐서·GPU 전송, infer = `net(x)[0]`, postprocess = clamp·CPU·uint8·BGR·잘라내기. registry에 `zerodcepp` 빌더 추가
- [ ] **Step 4: 통과 확인** — `pytest -q` → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 개선기 3단계 인터페이스, 단계별 시간, Zero-DCE++ 이름 정리"`

- [ ] **Step 6: Phase P1 기록** — `work_history.md`에 "Phase P1" 절을 이어서 추가 (바뀐 파일, 테스트 수, 결정)

---

# Phase P2: 탐지

### Task 4: BDD YOLO26 탐지기 (클래스 이름 매핑, 단계별 시간)

**Files:**
- Modify: `src/data/classes.py`, `src/detect/base.py`, `src/detect/yolo.py`
- Create: `scripts/download_detectors.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Produces:
  - `classes.BDD_MODEL_TO_COMMON: dict[str, int]` = `{"person": 0, "rider": 0, "bike": 1, "bicycle": 1, "car": 2, "motor": 3, "motorcycle": 3, "bus": 4, "truck": 5}`; 버리는 이름 `BDD_MODEL_IGNORED = {"train", "traffic light", "traffic sign"}`
  - `build_class_map(model_names: dict[int, str]) -> dict[int, int]` — 모델 id → 공통 id. 매핑에도 무시 목록에도 없는 이름이 있으면 `ValueError`(모르는 이름 목록 포함)
  - `Detections`에 `speed: dict[str, float]` 필드 추가 (기본 빈 dict, 키 `preprocess`, `inference`, `postprocess` ms — Ultralytics `Results.speed` 그대로)
  - `YoloDetector(weights: Path, imgsz: int, conf: float, device=0, name: str = "")` — 가중치의 `model.names`로 `build_class_map`. 속성 `name`, `imgsz`
  - `scripts/download_detectors.py`: 설정의 `detectors`마다 `hf_hub_download(hf_repo, "best.pt")`를 `weights` 경로로 복사. 이미 있으면 건너뜀
  - 기존 `coco_to_common`, `COCO_TO_COMMON`은 삭제

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_build_class_map_bdd`: `{0:"person",1:"rider",2:"car",3:"truck",4:"bus",5:"train",6:"motor",7:"bike",8:"traffic light",9:"traffic sign"}` → `{0:0,1:0,2:2,3:5,4:4,6:3,7:1}`
  - `test_build_class_map_unknown_raises`: `{0:"person",1:"scooter"}` → `ValueError`이고 메시지에 `scooter`
  - `test_apply_class_map`: 모델 cls `[5, 2, 8]`, 점수 `[.9,.8,.7]` → labels `[2]`, scores `[.8]` (train·traffic light 버림)
  - (가중치 있을 때만) `test_yolo_bdd_detect_returns_speed`: 검은 1280×720 이미지 → `Detections.speed`에 세 키
- [ ] **Step 2: 실패 확인** — `pytest tests/test_detect.py -v` → FAIL
- [ ] **Step 3: 구현** — `apply_class_map(boxes, cls, scores, class_map) -> Detections`를 순수 함수로 두고 `YoloDetector.detect`가 사용
- [ ] **Step 4: 통과 확인** — `pytest tests/test_detect.py -v` → PASS
- [ ] **Step 5: 사용자 PC에서 가중치 받기·확인** — `python -m scripts.download_detectors` 후 `python -c "from ultralytics import YOLO; print(YOLO('weights/bdd100k-yolo26s.pt').names)"` 출력이 위 10개 이름과 같은지 확인. 다르면 `BDD_MODEL_TO_COMMON`을 실제 이름에 맞추고 테스트도 고친다
- [ ] **Step 6: 커밋** — `git commit -m "feat: BDD100K YOLO26 탐지기, 클래스 이름 매핑, 단계별 시간"`

### Task 5: 여러 IoU 매칭, mAP50-95, 객체 크기별 지표

**Files:**
- Modify: `src/eval/matching.py`, `src/eval/metrics.py`
- Test: `tests/test_matching.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: 기존 `box_iou`
- Produces:
  - `IOU_THRS = np.round(np.arange(0.50, 0.96, 0.05), 2)` (10개)
  - `match_image_multi(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thrs=IOU_THRS) -> np.ndarray` — shape (N_pred, T), 매칭된 정답 인덱스, 없으면 -1. 임계값마다 기존 그리디 규칙(점수 내림차순, 같은 클래스, 정답 하나당 한 번)
  - `size_bucket(boxes: np.ndarray) -> np.ndarray` — 넓이로 `"small"`(< 32²), `"medium"`(32²~96²), `"large"`(> 96²) 문자열 배열
  - `ImageRecord(image_id, pred_boxes, pred_labels, pred_scores, match_idx, gt_boxes, gt_labels)` — `match_idx`는 (N, T)
  - `compute_metrics(records, eval_classes, conf_eval) -> list[dict]` — 행마다 `class`(클래스 이름 또는 `"all"`), `size`(`"all"`, `"small"`, `"medium"`, `"large"`), `precision`, `recall`, `f1`, `ap50`, `ap50_95`, `tp`, `fp`, `fn`, `n_gt`
    - P/R/F1은 IoU 0.5, conf ≥ conf_eval 기준
    - 크기 구간: 그 구간의 정답만 센다. 예측은 매칭된 정답이 그 구간이면 TP, 매칭이 없고 예측 박스 자체가 그 구간이면 FP, 매칭된 정답이 다른 구간이면 빼고 센다
    - `"all"` 클래스의 ap 값은 n_gt > 0인 클래스들의 평균. n_gt = 0이면 그 행의 ap는 NaN, 평균에서 뺀다
  - 기존 `average_precision`은 그대로 쓴다

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_match_multi_thresholds`: 정답 `[[0,0,10,10]]`, 예측 `[[0,0,10,8]]`(IoU 0.8) → 열 0~6(0.50~0.80)은 0, 열 7~9(0.85~0.95)는 -1
  - `test_ap50_95_perfect`: 예측 = 정답(점수 0.9) → ap50 = ap50_95 = 1.0
  - `test_ap50_95_partial`: 위 IoU 0.8 예측 하나 → ap50 = 1.0, ap50_95 = 0.7
  - `test_size_bucket`: 넓이 31², 32², 95², 97² → small, medium, medium, large
  - `test_empty_class_gives_nan_and_is_excluded`: car 정답만 있고 bus 정답 0 → bus 행 `n_gt == 0`, `math.isnan(ap50)`; `"all"` 행 ap50은 car의 ap50과 같음
  - `test_size_rows_exist`: 결과에 `(class="all", size)` 행이 4개(all, small, medium, large)
- [ ] **Step 2: 실패 확인** — `pytest tests/test_matching.py tests/test_metrics.py -v` → FAIL
- [ ] **Step 3: 구현** — 기존 `match_image`는 `match_image_multi(...)[:, 0] >= 0`으로 남겨 다른 곳 호환
- [ ] **Step 4: 통과 확인** — `pytest -q` → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: mAP50-95, 객체 크기별 지표"`
- [ ] **Step 6: Phase P2 기록** — `work_history.md`에 "Phase P2" 이어서 추가

---

# Phase P3: 외부 개선 모델

### Task 6: third_party 받기 스크립트와 불러오기 도우미

**Files:**
- Create: `scripts/fetch_third_party.py`, `third_party_manifest.json`, `src/enhance/third_party.py`
- Test: `tests/test_third_party.py`

**Interfaces:**
- Produces:
  - `third_party_manifest.json`: `{"zerodce": {"url": "https://github.com/Li-Chongyi/Zero-DCE", "commit": "<고정>"}, "sci": {"url": "https://github.com/vis-opt-group/SCI", ...}, "ruas": {"url": "https://github.com/KarelZhang/RUAS", ...}, "nightenh": {"url": "https://github.com/jinyeying/night-enhancement", ...}}`. commit은 Step 5에서 `git ls-remote <url> HEAD`로 얻은 값을 적는다
  - `scripts/fetch_third_party.py [--only NAME ...]`: 저장소마다 `third_party/<name>`에 clone 후 manifest의 commit으로 checkout. 이미 있고 commit이 같으면 건너뜀. Night-Enhancement 가중치(Dropbox)는 자동으로 받지 않고, 놓을 경로를 출력한다
  - `load_symbol(repo_dir: Path, module: str, attr: str) -> Any` — `repo_dir`(하위 폴더 포함 가능, 예: `third_party/sci/CVPR`)를 sys.path 맨 앞에 잠깐 넣고 `module`을 import해 `attr`을 돌려준다. **import 동안 새로 생긴 sys.modules 항목은 끝나면 지운다**(다음 저장소의 같은 이름 모듈과 섞이지 않게). 폴더나 모듈이 없으면 `EnhancerUnavailable`
  - `commit_of(repo_dir: Path) -> str | None` — `run_meta.json`에 남길 실제 commit

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_same_module_name_two_repos`: tmp에 `a/model.py`(`NAME = "A"`), `b/model.py`(`NAME = "B"`) → `load_symbol(a, "model", "NAME") == "A"` 다음 `load_symbol(b, "model", "NAME") == "B"`, 그리고 `"model" not in sys.modules`
  - `test_intra_repo_import`: `c/model.py`가 `from helper import X`, `c/helper.py`에 `X = 3` → `load_symbol(c, "model", "X") == 3`
  - `test_missing_repo_raises`: 없는 폴더 → `EnhancerUnavailable`
  - `test_manifest_has_four_repos`: manifest 키가 `{"zerodce", "sci", "ruas", "nightenh"}`이고 commit이 40자리 16진수
- [ ] **Step 2: 실패 확인** — `pytest tests/test_third_party.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `pytest tests/test_third_party.py -v` → PASS (manifest 테스트는 Step 5 뒤에 PASS)
- [ ] **Step 5: 사용자 PC에서 실행** — commit 값을 채우고 `python -m scripts.fetch_third_party` → `third_party/` 아래 4개 폴더 생성, `git status`에 third_party가 안 보이는지 확인
- [ ] **Step 6: 커밋** — `git commit -m "feat: third_party 저장소 받기, 모듈 충돌 없는 불러오기"`

### Task 7: Zero-DCE (원본) 래퍼

**Files:**
- Create: `src/enhance/zerodce_orig.py`
- Modify: `src/enhance/registry.py`
- Test: `tests/test_zerodce_orig.py`

**Interfaces:**
- Consumes: `load_symbol`, `assert_enhancer_contract`, `Enhancer` 3단계
- Produces: `ZeroDCEOrigEnhancer(repo_dir: Path, weights: Path, device=None)`, 조건 `zerodce`
  - 공식 `Zero-DCE_code/lowlight_test.py`와 같게: 네트워크 `model.enhance_net_nopool()` (`load_symbol(repo/"Zero-DCE_code", "model", "enhance_net_nopool")`), 가중치 `snapshots/Epoch99.pth`, 입력 RGB [0,1] (1,3,H,W), 출력은 반환값 3개 중 **두 번째**
  - Zero-DCE++의 `weights/Epoch99.pth`와 이름이 같지만 다른 파일이다 → 경로를 설정에서 따로 받는다

- [ ] **Step 1: 실패하는 테스트 작성** — 저장소·가중치가 있을 때만: `test_contract`(`assert_enhancer_contract`), `test_brightens_dark_image`(평균 밝기 20인 이미지 → 출력 평균 > 입력 평균); 없을 때: `test_missing_raises_unavailable`
- [ ] **Step 2: 실패 확인** — `pytest tests/test_zerodce_orig.py -v`
- [ ] **Step 3: 구현** — 가중치 로드는 `strict=True`, 실패하면 `EnhancerUnavailable`
- [ ] **Step 4: 통과 확인** (사용자 PC) — `pytest tests/test_zerodce_orig.py -v` → PASS, skip 아님
- [ ] **Step 5: 커밋** — `git commit -m "feat: Zero-DCE 원본 래퍼"`

### Task 8: SCI 래퍼

**Files:** Create `src/enhance/sci.py`; Modify `src/enhance/registry.py`; Test `tests/test_sci.py`

**Interfaces:**
- Produces: `SCIEnhancer(repo_dir: Path, variant: str = "difficult", device=None)`, 조건 `sci`
  - 공식 `CVPR/test.py`와 같게: `Finetunemodel(weights_path)` (`load_symbol(repo/"CVPR", "model", "Finetunemodel")`), 가중치 `CVPR/weights/{variant}.pt`, variant ∈ {easy, medium, difficult}
  - 입력 RGB [0,1] (1,3,H,W), 정규화 없음. `forward`는 `(i, r)`을 돌려주고 **`r`(입력 / 조명, [0,1]로 자른 값)이 개선 결과**
  - `Finetunemodel`은 가중치 키를 걸러서 로드하므로 키가 하나도 맞지 않아도 오류가 나지 않는다 → 래퍼에서 `torch.load`한 키와 모델 키가 **하나 이상 겹치고 모델 키가 전부 채워졌는지** 직접 확인하고, 아니면 `EnhancerUnavailable`

- [ ] **Step 1: 실패하는 테스트 작성** — 있을 때만 `test_contract`, `test_brightens_dark_image`, `test_all_model_keys_loaded`; 잘못된 variant(`"night"`) → `ValueError`; 없을 때 `test_missing_raises_unavailable`
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC, skip 아님)** — `pytest tests/test_sci.py -v`
- [ ] **Step 5: 커밋** — `git commit -m "feat: SCI 래퍼"`

### Task 9: RUAS 래퍼

**Files:** Create `src/enhance/ruas.py`; Modify `src/enhance/registry.py`; Test `tests/test_ruas.py`

**Interfaces:**
- Produces: `RUASEnhancer(repo_dir: Path, variant: str = "dark", device=None)`, 조건 `ruas`
  - 공식 `test.py`와 같게: `Network()` (`load_symbol(repo, "model", "Network")`), 가중치 `ckpt/{variant}.pt`, variant ∈ {lol, upe, dark}
  - `forward`는 `(u_list, r_list)`를 돌려준다. 결과는 **variant가 `lol`이면 `u_list[-1]`, `upe`·`dark`이면 `u_list[-2]`** (공식 test.py 그대로)
  - 입력: 공식 `multi_read_data.MemoryFriendlyLoader`의 test 변환을 읽고 같게 맞춘다(RGB [0,1]로 예상). 확인한 내용을 모듈 docstring에 적는다
  - 원래 PyTorch 0.4용 코드: import·로드 중 오류가 나면 원본을 고치지 말고 래퍼에서 처리한다(예: `Variable` 대신 일반 텐서 사용). 처리 내용을 docstring에 적는다

- [ ] **Step 1: 실패하는 테스트 작성** — 있을 때만 `test_contract`, `test_brightens_dark_image`, `test_variant_output_index`(lol은 -1, dark는 -2를 쓰는지 가짜 네트워크로 확인); 잘못된 variant → `ValueError`; 없을 때 `test_missing_raises_unavailable`
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC, skip 아님)** — `pytest tests/test_ruas.py -v`
- [ ] **Step 5: 커밋** — `git commit -m "feat: RUAS 래퍼"`

### Task 10: Night-Enhancement 래퍼

**Files:** Create `src/enhance/nightenh.py`; Modify `src/enhance/registry.py`; Test `tests/test_nightenh.py`

**Interfaces:**
- Produces: `NightEnhEnhancer(repo_dir: Path, variant: str = "delighteffects", device=None)`, 조건 `nightenh`
  - 가중치: variant `delighteffects` → `results/delighteffects/model/delighteffects_params_0600000.pt`, `lol` → `results/LOL/model/LOL_params_0900000.pt` (README 기준 위치, 저장소 폴더 아래)
  - **Step 0에서 정할 것:** 공식 `ENHANCENET.py`(`main.py`는 LOL, `main_delighteffects.py`는 빛 번짐 억제)의 test 단계에서 만드는 생성기 클래스와 생성자 인자, params 파일에서 읽는 키, 입력 변환(크기 조정·정규화), 생성기 호출과 출력 중 결과 인덱스, 출력 되돌리기(denormalize)를 읽고 래퍼를 그대로 맞춘다. 읽은 내용을 모듈 docstring에 적는다
  - 입력 크기 조건이 있으면(예: 배수) Zero-DCE++처럼 패딩 후 잘라내서 출력 크기를 입력과 같게 한다

- [ ] **Step 0: 공식 추론 코드 읽기** — 위 항목을 docstring 초안으로 적는다 (확인 불가 항목은 "미확인"으로 표시하고 사용자에게 알림)
- [ ] **Step 1: 실패하는 테스트 작성** — 있을 때만 `test_contract`, `test_loads_expected_key`; 없을 때 `test_missing_raises_unavailable`(가중치를 받을 위치를 메시지에 포함)
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC, skip 아님)** — `pytest tests/test_nightenh.py -v`
- [ ] **Step 5: 사용자 확인** — BDD 야간 이미지 3장 결과를 `results/check_nightenh/`에 저장해 사용자가 눈으로 확인 (헤드라이트 주변이 과하게 밝아지지 않는지)
- [ ] **Step 6: 커밋** — `git commit -m "feat: Night-Enhancement 래퍼"`
- [ ] **Step 7: Phase P3 기록** — `work_history.md`에 "Phase P3" 이어서 추가 (저장소별 commit, 확인한 추론 방식, 호환 처리)

---

# Phase P4: 새 데이터셋

공통: 로더는 `DetectionDataset`을 따르고 `load() -> list[Sample]`, 박스는 원본 해상도 xyxy(리사이즈는 Task 15의 러너에서 한다). 데이터를 받은 뒤 실제 라벨 파일 2개로 `tests/fixtures/`에 작은 사본을 만들어 테스트한다. **데이터가 아직 없으면 그 태스크는 건너뛰고 다음 페이즈로 간다** (러너가 없는 데이터셋을 건너뛰므로).

### Task 11: ACDC 야간 로더

**Files:** Create `src/data/acdc.py`; Modify `src/data/classes.py`; Test `tests/test_acdc.py`, `tests/fixtures/acdc_panoptic_mini.json`

**Interfaces:**
- Produces: `ACDCDataset(panoptic_json: Path, images_dir: Path, condition: str = "night")`, `name = "acdc_night"`, `eval_classes = [0,1,2,3,4,5]`
  - COCO 파노픽 형식 JSON(`images`, `annotations[].segments_info[]`의 `category_id`, `bbox`=xywh, `iscrowd`, `categories[].name`)을 읽는다. 클래스는 **이름으로** 매핑: `classes.ACDC_TO_COMMON = {"person": 0, "rider": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 4, "truck": 5}`. `iscrowd = 1`은 버린다
  - 이미지 경로는 JSON의 `file_name`을 `images_dir` 기준으로 찾는다. 실제 폴더 구조는 받은 데이터를 보고 맞춘다
- [ ] **Step 1: 실제 형식 확인** — 받은 ACDC의 파노픽 JSON 하나를 열어 키 이름이 위와 같은지 확인. 다르면 Interfaces를 고치고 work_history에 적는다
- [ ] **Step 2: 실패하는 테스트 작성** — 고정 JSON(이미지 2장, person·rider·car·train·crowd car 각 1개) → person 2개(person+rider), car 1개, train·crowd 없음, bbox xywh→xyxy
- [ ] **Step 3~5: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_acdc.py -v`
- [ ] **Step 6: 커밋** — `git commit -m "feat: ACDC 야간 로더"`

### Task 12: DarkDriving 로더 (밤 이미지 + 정렬된 낮 이미지)

**Files:** Create `src/data/darkdriving.py`; Modify `src/data/base.py`, `src/data/classes.py`; Test `tests/test_darkdriving.py`

**Interfaces:**
- Produces:
  - `Sample`에 선택 필드 `ref_path: Path | None = None` 추가 (정답 기반 화질 지표용 낮 이미지)
  - `DarkDrivingDataset(root: Path, split: str = "test")`, `name = "darkdriving"`, `eval_classes = [2]` (car만)
  - `classes.DARKDRIVING_TO_COMMON = {"car": 2, "Car": 2}`
- [ ] **Step 1: 실제 형식 확인** — 받은 테스트셋의 폴더 구조(밤·낮 이미지 쌍을 짓는 규칙)와 라벨 형식을 확인하고 docstring에 적는다
- [ ] **Step 2: 실패하는 테스트 작성** — 실제 라벨 2개로 만든 고정 파일 → 박스 개수·좌표, 모든 Sample의 `ref_path`가 존재하는 파일, 밤·낮 이미지 크기가 같음
- [ ] **Step 3~5: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_darkdriving.py -v`
- [ ] **Step 6: 커밋** — `git commit -m "feat: DarkDriving 로더"`

### Task 13: AI Hub 고정밀 야간 도심도로 로더

**Files:** Create `src/data/aihub_night.py`; Modify `src/data/classes.py`; Test `tests/test_aihub_night.py`

**Interfaces:**
- Produces: `AIHubNightDataset(labels_dir: Path, images_dir: Path)`, `name = "aihub_night"`, `eval_classes = [0,1,2,3,4,5]`
  - 정면 카메라 2D 파노픽 JSON에서 객체마다 폴리곤 점 목록 → 박스 = (min x, min y, max x, max y)
  - `classes.AIHUB_TO_COMMON = {"pedestrian": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 4, "truck": 5}`. "기타 차량", dontcare, 정적 객체는 버린다
  - 해상도는 JSON의 이미지 크기를 쓰지 않고 **실제 이미지 파일 크기**를 기준으로 한다 (페이지에 1920×1200과 1920×1080이 섞여 있음)
- [ ] **Step 1: 실제 형식 확인** — 받은 JSON의 키 이름(객체 목록, 클래스, 점)을 확인하고 docstring에 적는다. 클래스 이름이 다르면 매핑을 실제 이름으로 고친다
- [ ] **Step 2: 실패하는 테스트 작성** — 실제 라벨 2개로 만든 고정 파일 → 폴리곤 `[[10,20],[50,20],[50,80],[10,80]]` → 박스 `[10,20,50,80]`, 기타 차량 없음
- [ ] **Step 3~5: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_aihub_night.py -v`
- [ ] **Step 6: 커밋** — `git commit -m "feat: AI Hub 야간 도심도로 로더"`
- [ ] **Step 7: Phase P4 기록** — `work_history.md`에 "Phase P4" 이어서 추가 (받은 데이터셋, 실제 형식, 아직 없는 데이터셋)

---

# Phase P5: 화질 지표

### Task 14: pyiqa 화질 지표

**Files:** Create `src/eval/iqa.py`; Test `tests/test_iqa.py`

**Interfaces:**
- Produces:
  - `IQAEvaluator(metrics_nr: list[str], metrics_fr: list[str], device=0)` — 지표마다 `pyiqa.create_metric(name, device=...)`를 한 번만 만든다. pyiqa가 없으면 `IQAUnavailable`
  - `score(img_bgr: np.ndarray, ref_bgr: np.ndarray | None = None) -> dict[str, float]` — NR 지표는 항상, FR 지표는 `ref_bgr`가 있을 때만. 입력은 RGB [0,1] (1,3,H,W) 텐서로 바꾼다. FR은 두 이미지 크기가 다르면 `ValueError`
  - 값이 NaN·inf이거나 지표 계산이 예외를 내면 그 지표는 `float("nan")` (다른 지표는 계속)
  - `lower_better: dict[str, bool]` — 지표 이름 → `metric.lower_better` (분석과 그래프 방향에 사용)
  - `summarize_iqa(values: list[float]) -> dict` — `{"mean", "median", "n", "n_nan"}`, NaN은 평균·중앙값에서 뺀다

- [ ] **Step 1: 실패하는 테스트 작성**
  - (pyiqa 없어도) `test_summarize_iqa_ignores_nan`: `[1.0, nan, 3.0]` → mean 2.0, n 2, n_nan 1
  - (pyiqa 있을 때만) `test_fr_identical_images`: 같은 이미지 → psnr ≥ 40 또는 inf, ssim ≈ 1(오차 1e-3), lpips ≈ 0(오차 1e-3)
  - (pyiqa 있을 때만) `test_black_image_no_crash`: 전부 0인 256×256 → 예외 없이 dict, 값은 숫자 또는 NaN
  - (pyiqa 있을 때만) `test_fr_size_mismatch_raises`
  - (pyiqa 있을 때만) `test_lower_better_directions`: niqe·brisque·lpips는 True, musiq·clipiqa·psnr·ssim은 False
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인 (사용자 PC에서 pyiqa로)** — `pytest tests/test_iqa.py -v`
- [ ] **Step 5: 커밋** — `git commit -m "feat: pyiqa 화질 지표 7개"`
- [ ] **Step 6: Phase P5 기록** — `work_history.md`에 "Phase P5" 이어서 추가 (pyiqa 버전, 지표별 방향)

---

# Phase P6: 실험 러너와 분석

### Task 15: 실험 러너 개편

**Files:** Modify `scripts/run_experiment.py`; Test `tests/test_run_experiment.py`

**Interfaces:**
- Consumes: `resize_max_side`, `scale_boxes`, `Enhancer.enhance_timed`, `YoloDetector`, `match_image_multi`, `compute_metrics`, `IQAEvaluator`, `gpu_peak_mb`/`reset_gpu_peak`, 로더들
- Produces:
  - `build_datasets(cfg) -> tuple[list[DetectionDataset], list[dict]]` — 경로가 없는 데이터셋은 **오류 대신 건너뛰고** `{"dataset", "reason"}` 목록으로 돌려준다
  - `run_experiment(cfg, detectors=None, datasets=None, iqa=None, out_root=None, run_id=None) -> Path`
  - 결과 파일(스펙 5장): `detection_metrics.csv`(열: dataset, condition, detector, imgsz, class, size, precision, recall, f1, ap50, ap50_95, tp, fp, fn, n_gt), `iqa_metrics.csv`(dataset, condition, 지표마다 `<m>_mean`, `<m>_median`, `<m>_n_nan`), `timing.csv`(dataset, condition, `enh_pre_ms`/`enh_infer_ms`/`enh_post_ms`/`enh_total_ms` 각 mean·median·p95, 탐지 설정마다 `det_<detector>_<imgsz>_ms` mean·median, `gpu_peak_mb`), `per_image.jsonl`, `run_meta.json`(설정, 환경, 외부 저장소 commit, `skipped_datasets`, `skipped_conditions`, `excluded_images`, `sample_ids`), `samples/<dataset>/<condition>/*.jpg`
  - 흐름: 데이터셋마다 표본 추출 → 이미지 읽기·리사이즈(박스 같이) → 조건마다 워밍업 10회 → 이미지마다 `enhance_timed` 1회 → IQA(DarkDriving은 리사이즈한 `ref_path` 이미지로 FR 포함) → 탐지 설정 4개 각각 탐지·매칭
  - **이미지 제외 규칙:** 어떤 조건에서든 한 이미지가 실패하면(예: GPU 메모리 부족), 그 이미지를 **그 데이터셋의 모든 조건 집계에서** 뺀다. 그래서 집계는 모든 조건을 다 돈 뒤 마지막에 한다
  - 조건마다 시작할 때 `reset_gpu_peak()`, 끝날 때 `gpu_peak_mb()`
  - GPU 메모리 부족(`torch.cuda.OutOfMemoryError`)이면 `torch.cuda.empty_cache()` 후 한 번 다시 시도하고, 또 실패하면 위 제외 규칙을 따른다. (스펙 8장의 "타일 처리 재시도"는 쓰지 않는다: 긴 변 1280 제한으로 메모리 부족 가능성이 낮고, 곡선·조명 추정 모델은 타일 경계에서 결과가 달라진다. 이 변경을 work_history에 기록)
  - CLI 옵션 `--check`: 데이터셋 경로·탐지 가중치·third_party 폴더와 가중치·pyiqa·CUDA 상태를 표로 출력하고 실험 없이 끝낸다

- [ ] **Step 1: 실패하는 테스트 작성** (가짜 탐지기 2개 `name`이 다른 OracleDetector, `imgsz` 속성 포함, 가짜 IQA)
  - `test_smoke_outputs`: 6개 결과 파일 존재, `detection_metrics.csv`에 `(condition="original", class="all", size="all")` 행이 탐지 설정 수(2모델 × 2크기 = 4)만큼, 그 f1 = 1.0
  - `test_missing_dataset_skipped`: 경로가 없는 데이터셋 설정 → 오류 없이 끝나고 `run_meta.json`의 `skipped_datasets`에 이름
  - `test_failure_excludes_image_from_all_conditions`: 한 이미지에서만 예외를 내는 가짜 개선기 → `original` 집계의 이미지 수도 하나 줄어 있음, `excluded_images`에 기록
  - `test_resize_applied`: 2000×3000 가짜 이미지 → 탐지기가 받은 이미지의 긴 변이 1280, 정답 박스도 같은 비율
  - `test_fr_only_with_ref`: `ref_path` 있는 데이터셋만 `iqa_metrics.csv`의 psnr 열에 값, 나머지는 빈 값
  - `test_no_absolute_user_paths`: `run_meta.json` 텍스트에 홈 디렉터리 경로가 없음
  - `test_oom_retry_once`: 첫 호출만 `OutOfMemoryError`를 내는 가짜 개선기 → 이미지가 제외되지 않음
  - `test_check_mode_runs_nothing`: `main(["--check"])` → 결과 폴더가 생기지 않음
- [ ] **Step 2: 실패 확인** — `pytest tests/test_run_experiment.py -v` → FAIL
- [ ] **Step 3: 구현** — 기존 `draw_boxes`, `_safe_name`, `WarmupFailed` 재사용
- [ ] **Step 4: 통과 확인** — `pytest -q` → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 재설계 실험 러너 (탐지 4설정, 화질 지표, 단계별 시간)"`

### Task 16: 분석 (변화량, 상관관계, 빠른/확실한 모델)

**Files:** Create `src/eval/analysis.py`, `scripts/analyze.py`; Test `tests/test_analysis.py`

**Interfaces:**
- Consumes: Task 15의 CSV 열 이름
- Produces:
  - `delta_table(det: pd.DataFrame) -> pd.DataFrame` — `class="all"`, `size="all"` 행에서 (dataset, detector, imgsz)마다 각 조건의 값 − `original` 값. 열 `d_precision`, `d_recall`, `d_f1`, `d_ap50`, `d_ap50_95`
  - `iqa_correlation(iqa: pd.DataFrame, delta: pd.DataFrame, detector="yolo26s_bdd", imgsz=640) -> pd.DataFrame` — 데이터셋 × 지표마다 조건별 `<m>_mean`과 `d_ap50`의 스피어만 상관계수(`pandas.Series.corr(method="spearman")`)와 조건 수. 지표 방향은 그대로 두고(높을수록 좋은지 여부를 열 `lower_better`로 함께 남김)
  - `select_models(delta, timing, budget_ms=33.3, detector="yolo26s_bdd", imgsz=640) -> dict` — `{"fast": 조건 이름 또는 None, "reliable": 조건 이름, "table": DataFrame}`
    - 합계 시간 = `enh_total_ms_mean` + `det_<detector>_<imgsz>_ms_mean`, 데이터셋 평균
    - fast: `original` 제외, 합계 ≤ budget_ms 중 데이터셋 평균 `d_ap50` 최대. 없으면 None
    - reliable: `original` 제외, 데이터셋마다 `d_ap50` 순위(1 = 최고) → 순위 평균이 가장 작은 조건. 동점이면 평균 `d_ap50`이 큰 쪽
  - `scripts/analyze.py <run_dir>` — `analysis/` 아래 `delta.csv`, `iqa_correlation.csv`, `selection.json`, 그림 `speed_vs_dmap.png`(x 합계 ms, y d_ap50, 33.3ms 세로선), `dmap_by_dataset.png`, 그리고 나머지 3개 탐지 설정에서의 선택 결과 `selection_other_settings.csv`
- [ ] **Step 1: 실패하는 테스트 작성** — 손으로 만든 작은 DataFrame으로
  - `test_delta_original_zero`: original 행의 d_ap50 = 0
  - `test_select_fast_respects_budget`: 조건 A(20ms, +0.02), B(50ms, +0.05) → fast = A, reliable = B
  - `test_select_fast_none_when_all_slow`: 모두 40ms → fast is None
  - `test_reliable_uses_rank_mean`: 데이터셋 2개에서 A는 (1위, 3위), B는 (2위, 2위), C는 (3위, 1위) → 순위 평균 동점 2.0, 평균 d_ap50로 결정되는지
  - `test_spearman_perfect`: 지표 평균과 d_ap50이 같은 순서 → 1.0
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_analysis.py -v`
- [ ] **Step 5: 커밋** — `git commit -m "feat: 결과 분석과 빠른/확실한 모델 선택"`
- [ ] **Step 6: Phase P6 기록** — `work_history.md`에 "Phase P6" 이어서 추가

---

# Phase P7: 전체 실험 (사용자 PC)

### Task 17: 준비 확인과 전체 실행

**Files:** Modify `README.md`(실행 순서), `work_history.md`

- [ ] **Step 1: 준비 확인** — `python -m scripts.run_experiment --check` (Task 15). 빠진 것이 있으면 사용자에게 알리고 그 항목은 건너뛴 채 진행할지 묻는다
- [ ] **Step 2: 작은 실행** — `--num-samples 5`로 전체 흐름 확인. 결과 폴더의 6개 파일, `samples/` 이미지 확인
- [ ] **Step 3: 전체 실행** — 기본 300장. 걸린 시간과 건너뛴 항목 기록
- [ ] **Step 4: 분석 실행** — `python -m scripts.analyze results/<run_id>`
- [ ] **Step 5: 시간 측정 확인** — Zero-DCE++의 `enh_infer_ms`가 논문 값(1080p에서 1.6ms, Titan RTX)과 크게 다르면(10배 이상) 원인을 찾아 기록한다 (예: cudnn benchmark, 첫 호출, 해상도)
- [ ] **Step 6: Phase P7 기록** — `work_history.md`에 결과 요약(데이터셋별 장수, 조건별 주요 수치, 빠른/확실한 모델) 이어서 추가

---

# Phase P8: 결과물

### Task 18: 영상 프레임 → 비교 이미지

**Files:** Create `src/frames.py`, `src/visualize.py`, `scripts/compare_frame.py`; Test `tests/test_frames.py`, `tests/test_visualize.py`

**Interfaces:**
- Produces:
  - `extract_frame(video: Path, frame_index: int | None = None, time_s: float | None = None) -> np.ndarray` — 둘 중 하나만 받는다. 범위를 벗어나면 `ValueError`(영상 길이 포함)
  - `draw_detections(img, det: Detections, conf_eval: float, gt_boxes=None) -> np.ndarray` — 예측 초록, 정답 빨강, 클래스 이름 표시
  - `make_grid(tiles: list[tuple[str, np.ndarray]], cols: int = 4) -> np.ndarray` — 각 타일 위에 조건 이름과 개선 ms를 쓴다
  - `scripts/compare_frame.py --input <영상 또는 이미지> [--frame N | --time S] --detector yolo26s_bdd --imgsz 640 --out <png>` — 조건 8개로 개선·탐지해 격자 이미지 저장
- [ ] **Step 1: 실패하는 테스트 작성** — tmp에 OpenCV로 10프레임 영상(프레임마다 밝기 다르게) → `frame_index=3`의 평균 밝기가 그 프레임 값, `frame_index=10` → `ValueError`; `make_grid` 타일 5개·cols 4 → 높이가 타일 2줄
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_frames.py tests/test_visualize.py -v`
- [ ] **Step 5: 커밋** — `git commit -m "feat: 영상 프레임 비교 이미지"`

### Task 19: Streamlit 화면

**Files:** Create `app/streamlit_app.py`, `app/views.py`; Test `tests/test_views.py`

**Interfaces:**
- Consumes: Task 15·16 결과 파일, Task 18 함수
- Produces:
  - `views.py`(Streamlit 없이 테스트 가능): `list_runs(root: Path) -> list[str]`(최신순), `load_run(run_dir) -> dict[str, pd.DataFrame]`(detection, iqa, timing, delta, correlation), `summary_table(det, detector, imgsz) -> pd.DataFrame`
  - 화면 탭 3개: (1) 결과 — 탐지 설정 선택, 조건별 지표표·Δ 막대그래프·속도-ΔmAP 그래프·빠른/확실한 모델 표시, 화질 지표표·상관표 (2) 비교 이미지 — `samples/` 보기 (3) 데모 — 이미지나 영상(+프레임 번호) 올리기 → `make_grid` 결과
  - 실행: `streamlit run app/streamlit_app.py`
- [ ] **Step 1: 실패하는 테스트 작성** — 가짜 결과 폴더 2개 → `list_runs` 최신순, `summary_table`이 `class="all"`·`size="all"` 행만
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인** — `pytest tests/test_views.py -v`
- [ ] **Step 5: 사용자 PC에서 화면 확인** — 탭 3개가 오류 없이 열림
- [ ] **Step 6: 커밋** — `git commit -m "feat: Streamlit 결과 화면"`

### Task 20: 결과 문서

**Files:** Create `RESULTS_enhancement.md`; Modify `README.md`, `work_history.md`

- [ ] **Step 1: 작성** — 실험 설정(모델·데이터셋·지표와 출처), 데이터셋별 결과표, 처리 시간, 화질 지표와 상관관계, 빠른/확실한 모델과 근거, 한계(예: 정답 없는 지표의 한계, CLIP-LIT 등 제외 이유, 가중치 변형 선택 이유), 범위 변경(비교 영상 → 비교 이미지). 수치는 모두 결과 CSV에서 옮긴다
- [ ] **Step 2: 확인** — 문서의 수치를 CSV와 대조하는 짧은 스크립트로 확인 (불일치 0개)
- [ ] **Step 3: 커밋** — `git commit -m "docs: 재설계 실험 결과"`
- [ ] **Step 4: Phase P8 기록** — `work_history.md`에 "Phase P8"과 최종 요약 이어서 추가
