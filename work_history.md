# 작업 기록 (work_history)

> 이 파일은 아래로 추가만 합니다. 기존 내용은 수정·삭제하지 않습니다.

---

## 2026-11-10 — 설계와 계획

- 제안서(hwpx) 분석 → 설계 결정: 구조 A(배치 실험 + 결과 뷰어), YOLO26, 기법 4종(Gamma/CLAHE/Zero-DCE++/Real-ESRGAN), 데이터셋 BDD100K 야간 + ExDark(AI Hub는 보류)
- 스펙: `docs/superpowers/specs/2026-11-10-lowlight-detection-design.md`
- 계획: `docs/superpowers/plans/2026-11-10-lowlight-detection.md` (Phase 1~4, Task 1~14)

---

## Phase 1 (11주차): 골격 · 데이터 · 지표 · 기본 기법 · 실험 러너

### Task 1: 프로젝트 골격과 설정 로더 — 완료
- 추가: `configs/default.yaml`, `src/config.py`(load_config, PROJECT_ROOT, ConfigError), `requirements.txt`, `.gitignore`, `README.md`, `pytest.ini`
- 상대 경로 키(labels_json, images_dir, annos_dir, weights, root)는 프로젝트 루트 기준 절대 경로로 변환
- 테스트: `tests/test_config.py` 3개 통과

### Task 2: 공통 클래스 · Sample · BDD100K 로더 — 완료
- 추가: `src/data/classes.py`(COMMON_CLASSES, COCO/BDD/ExDark 매핑), `src/data/base.py`(Sample, DetectionDataset, select_samples), `src/data/bdd100k.py`
- BDD 2018 라벨(person/bike/motor)과 det_20 라벨(pedestrian/bicycle/motorcycle)을 모두 같은 클래스로 매핑
- `labels` 키가 없는 이미지는 GT 0개로 처리, box2d 없는 라벨(차선·주행영역)은 무시
- 표본 추출은 image_id 정렬 후 seed 고정 → 입력 순서와 무관하게 동일
- 테스트: `tests/test_bdd100k.py` 7개 통과

### Task 3: ExDark 로더 — 완료
- 추가: `src/data/exdark.py`(ExDarkDataset, parse_bbgt)
- bbGt `l t w h` → xyxy 변환, People/Bicycle/Car/Motorbike/Bus만 사용
- 라벨 파일은 소문자 이름으로 색인해서 `a.JPG` ↔ `a.jpg.txt` 같은 대소문자 차이를 흡수
- 라벨 파일이 없거나 평가 클래스 박스가 0개인 이미지는 제외
- 결정: image_id는 `<클래스 폴더>/<파일명>` (폴더가 달라도 id가 겹치지 않게)
- 테스트: `tests/test_exdark.py` 5개 통과

### Task 4: 박스 매칭과 지표 — 완료
- 추가: `src/eval/matching.py`(box_iou, match_image), `src/eval/metrics.py`(ImageRecord, filter_to_classes, average_precision, compute_metrics)
- 매칭: 같은 클래스 + IoU ≥ 0.5, 점수 내림차순 그리디, 정답 1개당 1번
- P/R/F1은 conf ≥ 0.25 예측만, AP@0.5는 전체 점수 범위(all-point 보간). `all`은 micro P/R/F1 + GT 있는 클래스의 mAP
- 평가 클래스 밖의 예측(예: ExDark에서 truck)은 FP로 세지 않음. GT 없는 이미지의 예측은 전부 FP
- 테스트: `tests/test_matching.py` 9개, `tests/test_metrics.py` 9개 통과 (AP 손계산 값 0.8333 검증 포함)

### Task 5: Enhancer 인터페이스 · Gamma · CLAHE · 레지스트리 — 완료
- 추가: `src/enhance/base.py`(Enhancer, IdentityEnhancer, EnhancerUnavailable), `gamma.py`(LUT 방식), `clahe.py`(LAB의 L 채널만), `registry.py`(build_enhancers)
- 설정의 sweep 값마다 조건 1개 생성 → 현재 7개 조건(original, gamma 3, clahe 3)
- Zero-DCE++, Real-ESRGAN은 Phase 2 전까지 "not implemented yet" 사유로 skipped 목록에 들어감
- 테스트: `tests/test_enhance.py` 9개 통과

### Task 6: 탐지기 래퍼 (YOLO26) — 완료 (GPU 스모크 테스트는 사용자 PC에서)
- 추가: `src/detect/base.py`(Detections, Detector), `src/detect/yolo.py`(coco_to_common, YoloDetector)
- COCO 클래스 중 person/bicycle/car/motorcycle/bus/truck만 남기고 공통 id로 변환
- CUDA가 없으면 경고 후 CPU로 실행
- 결정: `weights/yolo26s.pt` 파일이 없으면 이름만 넘겨 ultralytics가 공식 가중치를 자동 다운로드
- 테스트: `tests/test_detect.py` 2개 통과, `test_yolo_smoke`는 ultralytics 없는 환경이라 skip (사용자 PC에서 실행)

### Task 7: 시간 측정 · 결과 저장 · 실험 러너 — 완료
- 추가: `src/timing.py`(timed, summarize_ms, cuda_sync), `src/io_utils.py`(CSV/JSON/JSONL 저장, collect_env), `scripts/run_experiment.py`
- 결과 폴더 `results/<run_id>/`: metrics.csv, timing.csv, per_image.jsonl, run_meta.json, samples/
- 데이터셋 경로가 없으면 실험 시작 전에 경로를 알려 주고 중단. 읽을 수 없는 이미지와 처리 중 오류가 난 이미지는 `excluded_images`에 기록하고 계속 진행
- 화질 개선 시간과 탐지 시간을 따로 측정, 조건마다 워밍업, GPU 사용 시 CUDA 동기화
- CSV는 utf-8-sig로 저장해 엑셀에서 한글이 깨지지 않음
- 테스트: `tests/test_timing.py` 4개, `tests/test_run_experiment.py` 3개 통과

### Phase 1 최종 점검 — 수정 1건
- 발견: Windows의 `cv2.imread`/`cv2.imwrite`는 한글 경로(`바탕 화면`)를 처리하지 못해, 그대로 두면 모든 이미지가 "읽을 수 없음"으로 제외됨
- 수정: `src/imageio.py`(read_image, write_image) 추가 — 바이트로 읽고 `cv2.imdecode`/`imencode`로 처리. 러너가 이 함수를 쓰도록 변경
- 테스트: `test_unicode_path_images_still_read` (cv2.imread가 실패하는 상황을 흉내) 실패 확인 → 수정 후 통과
- 전체 테스트: 52개 통과, 1개 skip(`test_yolo_smoke`, ultralytics 필요)

### Phase 1 요약
- 완료: Task 1~7. Gamma 3 + CLAHE 3 + 원본 = 7개 조건으로 실험 가능
- 남은 확인(사용자 PC): `pip install -r requirements.txt` → `pytest` → 데이터 배치 후 `python -m scripts.run_experiment --num-samples 20`
- 다음: Phase 2 (Zero-DCE++, Real-ESRGAN)

---

## Phase 2 (12주차): 딥러닝 화질 개선 기법

### Task 8: Zero-DCE++ — 코드 완료, 실제 모델 실행 확인은 사용자 PC에서
- 추가: `src/enhance/zerodce.py`(ZeroDCEPPNet, ZeroDCEEnhancer, pad_to_multiple), `registry.py`에 연결
- 네트워크는 공식 구현과 같은 모듈 이름(e_conv1~7, depth_conv/point_conv)으로 작성 → 공식 Epoch99.pth를 그대로 로드. 파라미터 수 계산값 10,561개로 논문의 "약 10K"와 일치(PC 테스트로 확인 예정)
- 가중치 구조가 맞지 않으면 실험을 멈추지 않고 이유를 남기고 건너뜀
- 결정: H·W가 scale_factor 배수가 아니면 공식 코드처럼 잘라내지 않고 가장자리로 채운 뒤 원래 크기로 되돌림 (탐지기 입력 크기를 모든 조건에서 같게 유지하기 위해)
- 결정: scale_factor는 스펙대로 1 (논문 기본값은 12로 더 빠르지만 품질이 낮을 수 있음. 설정에서 바꿀 수 있음)
- 테스트: 여기서 4개 통과, PyTorch가 필요한 3개(출력 크기, scale 12, 파라미터 수)는 PC에서 실행

### Task 9: Real-ESRGAN — 코드 완료, 실제 모델 실행 확인은 사용자 PC에서
- 추가: `src/enhance/realesrgan.py`(RealESRGANEnhancer, run_with_tile_retry, patch_torchvision_functional_tensor), `registry.py`에 연결
- RealESRGAN_x2plus(RRDBNet 23블록)로 2배 업스케일 → `cv2.INTER_AREA`로 원래 크기 복원
- GPU 메모리 부족 시 타일 400 → 200 → … 로 줄여 재시도(최소 128), 줄인 값은 다음 이미지에도 유지
- 호환 패치: torchvision 0.17부터 `functional_tensor` 모듈이 없어져 basicsr import가 실패하는 알려진 문제 → `functional`을 그 이름으로 등록
- 테스트: 여기서 6개 통과(타일 재시도 3, 호환 패치 1, 가중치·패키지 없음 2), 실제 업스케일 1개는 PC에서

### Phase 2 점검 — 수정 1건 (Phase 1에서 미뤄 둔 항목)
- 문제: 어떤 기법이 워밍업 단계에서 오류를 내면(예: GPU 오류, 가중치 문제) 실험 전체가 멈춤
- 수정: 워밍업 실패 시 그 데이터셋의 그 조건만 건너뛰고, `skipped_conditions`에 데이터셋 이름과 오류 내용을 기록한 뒤 다음 조건으로 진행
- 테스트: `test_condition_failing_at_warmup_is_skipped` 실패 확인 → 수정 후 통과

### Task 10: 가중치 안내 — 문서 완료, 전체 실험은 사용자 PC에서
- 추가: `scripts/download_weights.md`(가중치 3종의 공식 출처와 확인 방법), README 가중치 항목 갱신
- `configs/default.yaml`의 scale_factor 옆에 논문 기본값(12) 메모
- 전체 테스트: 63개 통과, 5개 skip(PyTorch·realesrgan·ultralytics가 필요한 테스트 → PC에서 실행)

### Phase 2 요약
- 완료: 9개 조건(원본, Gamma 3, CLAHE 3, Zero-DCE++, Real-ESRGAN) 실험 코드
- 남은 확인(사용자 PC): 가중치 2개 받기 → `pytest` 전체 통과 확인(skip 0개 목표) → `python -m scripts.run_experiment` 전체 실행 → `skipped_conditions`가 비어 있는지 확인
- 다음: Phase 3 (비교 영상, Streamlit 웹 화면)

### 수정: requirements.txt 설치 오류 (Windows)
- 문제: `pip install -r requirements.txt`가 `UnicodeDecodeError: 'cp949'`로 실패. 한국어 Windows의 pip가 파일을 cp949로 읽는데 한글 주석이 UTF-8로 저장돼 있었음
- 수정: requirements.txt 주석을 영어로 바꿔 ASCII만 남김
- 테스트: `tests/test_project_files.py::test_requirements_is_ascii` 실패 확인 → 수정 후 통과. 전체 64개 통과, 5개 skip

### 수정: Real-ESRGAN 의존성 교체 (basicsr 설치 실패)
- 문제: `pip install -r requirements.txt`가 basicsr 빌드 단계에서 `KeyError: '__version__'`로 실패. basicsr의 setup.py가 exec()로 지역 변수를 채우는 방식인데, Python 3.13부터(PEP 667) 이 방식이 동작하지 않음. pip는 하나가 실패하면 전체 설치를 멈추므로 다른 라이브러리도 설치되지 않았음
- 결정: Python 버전을 낮추는 대신 realesrgan·basicsr를 빼고 `spandrel`로 공식 RealESRGAN_x2plus.pth를 불러오도록 변경 (가중치 파일은 그대로)
- 변경: `src/enhance/realesrgan.py` — spandrel로 모델 로드, 타일 분할 처리(`tile_process`)를 직접 구현, 메모리 부족 시 타일 축소 재시도는 유지. torchvision 호환 패치는 basicsr를 안 쓰므로 삭제
- 변경: `requirements.txt` — realesrgan, basicsr 삭제, spandrel 추가
- 테스트: `test_tile_process_matches_whole_image`(타일로 나눠도 결과가 같은지), `test_tile_process_tile_zero_runs_whole_image`, `test_requirements_installable_on_python_313` 추가 → 실패 확인 후 통과. 전체 66개 통과, 5개 skip

### 변경: Real-ESRGAN을 공식 패키지로 되돌림 (사용자 요청)
- 배경: 사용자 PC를 Python 3.11.9 가상환경으로 새로 구성 → basicsr 설치 가능 (실패 원인은 Python 3.13+였음)
- 사용자 요청에 따라 spandrel을 빼고 공식 `realesrgan` + `basicsr`로 복귀
- 변경: `requirements.txt`(spandrel 삭제, realesrgan·basicsr 추가, Python 3.11/3.12 필요 명시), `src/enhance/realesrgan.py`(공식 RealESRGANer 사용, torchvision 0.17+ 호환 패치 복구, 타일 축소 재시도 유지), README 설치 안내(Python 3.11/3.12, Git Bash 명령)
- 테스트: `test_requirements_use_official_realesrgan`, `test_patch_functional_tensor` 실패 확인 → 수정 후 통과. 전체 65개 통과, 5개 skip

### 추가: BDD100K 다운로드·변환 스크립트 (Hugging Face 사본)
- 배경: 공식 BDD100K는 가입이 필요하고 내 쪽 브라우저에서는 공식 사이트가 열리지 않음 → 사용자 결정으로 Hugging Face 비공식 사본(`dgural/bdd100k`, 검증 세트, 약 695MB, FiftyOne 형식) 사용. BDD100K 라이선스(교육·연구·비영리) 적용
- Claude 작업 환경(클라우드·PC 연결 셸)은 외부 다운로드가 막혀 있어(403) 다운로드는 사용자 PC에서 실행
- 추가: `scripts/download_bdd100k_hf.py` — `huggingface_hub`로 받은 뒤 FiftyOne `samples.json`(상대 좌표 [x,y,w,h])을 BDD 라벨 형식 `bdd_labels.json`(픽셀 xyxy, timeofday 포함)으로 변환. 이미지 크기 정보가 없으면 이미지를 읽어서 계산
- 변경: `configs/default.yaml` bdd100k 기본 경로 → `datasets/bdd100k_hf` (공식 경로는 주석으로 남김), `requirements.txt`에 huggingface_hub 추가, README 데이터 배치 갱신
- 테스트: `tests/test_bdd_hf.py` 5개(좌표 변환, 크기 정보 없음, 절대 경로, 검출 없음, 기존 BDD 로더로 읽기) + 설정·requirements 테스트 → 실패 확인 후 통과. 전체 71개 통과, 5개 skip

### 수정: BDD100K 다운로드 속도 개선
- 문제: 저장소 파일 20,006개를 하나씩 받아 예상 1시간 이상 소요
- 수정: 변환에 필요한 `samples.json`, `metadata.json`, `fiftyone.yml`, `data/*`만 받도록 제한(`fields/` 제외), 동시 다운로드 8 → 32개. 다시 실행하면 이미 받은 파일은 건너뜀
- 테스트: `test_download_skips_unneeded_files` 실패 확인 → 통과. 전체 72개 통과, 5개 skip

### 사용자 PC 첫 테스트 결과와 수정
- PC(Python 3.11.9, RTX 3070)에서 `pytest`: 73개 통과, 2개 실패, 2개 skip(패키지가 없을 때만 도는 테스트라 정상)
- 확인된 것: 스모크 테스트 안에서 Zero-DCE++ 공식 가중치(Epoch99.pth)가 strict 로드에 성공하고 Real-ESRGAN도 실제로 실행됨 → Zero-DCE++ 네트워크 구조가 공식과 일치함을 확인
- 실패 원인: `test_build_enhancers_expands_sweep`, `test_smoke_run`이 "가중치가 없어 딥러닝 기법은 건너뛴다"는 전제로 작성돼 있었는데, PC에는 가중치가 있어 조건이 2개 더 생김 (코드 문제 아님, 테스트가 환경에 의존)
- 수정: 두 테스트에서 딥러닝 기법의 가중치 경로를 일부러 없는 경로로 바꿔, 가중치 유무와 상관없이 같은 결과가 나오게 함
- 수정: 공식 RealESRGANer가 이미지마다 출력하는 `Tile 1/1` 메시지를 숨김 (PC 로그에서 확인됨). 테스트 `test_no_tile_progress_output` 추가(PC에서 실행)
- 데이터 확인: BDD100K HF 사본 변환 결과 전체 10,000장, 야간 3,929장 — 논문에 나온 공식 검증 세트 야간 수(3,929)와 일치

### 개인정보 제거 (사용자 요청)
- 점검 범위: 프로젝트 폴더 전체(.venv, datasets, weights 제외)에서 팀원 이름, Windows 사용자 이름, 이메일, 사용자 폴더 경로 검색
- 설계 문서: 팀원 이름과 사용자 이름이 들어간 작업 폴더 경로 삭제
- 실험 결과: `run_meta.json`에 설정 경로가 절대 경로(사용자 이름 포함)로 저장되던 문제 수정 → 프로젝트 안 경로는 상대 경로, 홈 폴더 경로는 `~`로 저장. 건너뜀 사유 같은 문장 속 경로도 같은 방식으로 가림 (`src/io_utils.py`의 `relativize_paths`)
- `.gitignore`: 로그 파일(`*.log`, `pytest_log.txt`)과 가중치(`*.pt`, `*.pth`, 루트의 yolo26s.pt 포함) 추가
- 테스트: `tests/test_io_utils.py` 4개, `test_run_meta_has_no_absolute_project_paths`, `test_gitignore_excludes_logs_and_weights` 실패 확인 → 통과. 전체 78개 통과, 6개 skip
- 남은 것(사용자 결정 필요): 이미 GitHub에 커밋된 `pytest_log.txt`에 사용자 경로가 들어 있음. 파일 삭제와 저장소 기록 정리는 사용자 확인 후 진행
- 추가(사용자 요청): README 첫 줄의 팀 번호("10조") 삭제
- 변경(사용자 요청): README를 처음 보는 사람이 따라 하기 쉽게 다시 작성 — 프로젝트 목적, 비교 기법 표, 공정 비교 조건, 데이터셋, 설치·데이터·가중치 준비, 실행 명령, 결과 파일 설명, 폴더 구조, 일정, 출처·라이선스 순서로 정리

### 첫 실험 결과 (BDD100K 야간 20장, run 20261111-005719)
- 환경: Python 3.11.9, torch 2.11.0+cu128, ultralytics 8.4.208, RTX 3070. 9개 조건 모두 실행(건너뜀 0, 제외 이미지 0)
- 전체(all) F1 / AP50: 원본 0.377 / 0.326, Gamma1.5 0.393 / 0.322, Gamma2.0 0.349 / 0.280, Gamma2.5 0.355 / 0.255, CLAHE2.0 0.384 / 0.372, CLAHE3.0 0.383 / 0.361, CLAHE4.0 0.377 / 0.332, Zero-DCE++ 0.321 / 0.280, Real-ESRGAN 0.373 / 0.330
- 화질 개선 시간(평균 ms/장): Gamma 0.8, CLAHE 11, Zero-DCE++ 47, Real-ESRGAN 905. 탐지 시간은 모든 조건 12~15ms
- 해석 시 주의: 정답 박스 210개(car 179, person 25)뿐이라 조건 간 차이가 TP 1~4개 수준 → 통계적으로 의미 있다고 보기 어려움. 300장 실험 필요
- 관찰: 모든 조건에서 Recall이 약 0.25로 낮음. 정답 박스의 45%가 원본 기준 32×32px 미만(입력 640으로 줄이면 16×16 미만)인 작은 객체라, 밝기보다 객체 크기가 Recall을 제한하는 것으로 보임
- 샘플 이미지로 정답 박스 위치가 이미지와 맞는 것 확인. Zero-DCE++는 밝아지지만 노이즈도 커지고 FP가 18→25로 증가

### 추가: ExDark 다운로드·정리 스크립트
- 출처 확인: 공식 GitHub의 `Dataset/README.md`(이미지, Google Drive, 약 1.5GB)와 `Groundtruth/README.md`(라벨, Google Drive, 약 4.2MB). 라벨 형식은 `% bbGt` 헤더 + `클래스 l t w h ...`로 기존 로더와 일치. 이용 조건: 비상업적 학술 연구·교육
- Claude 작업 환경은 외부 다운로드가 막혀 있어 사용자 PC에서 실행하는 스크립트로 작성
- 추가: `scripts/download_exdark.py` — gdown으로 두 zip을 받아 압축 해제 → 압축 안 폴더 이름과 상관없이 클래스 폴더(Car, People 등)를 찾아 `datasets/ExDark/ExDark`, `datasets/ExDark/ExDark_Annno`로 정리. 이미 받은 zip은 다시 받지 않음. Google Drive 제한으로 실패하면 직접 받은 zip을 `--images-zip`, `--annos-zip`으로 넘길 수 있음
- 변경: `requirements.txt`에 gdown 추가, README ExDark 안내 갱신
- 테스트: `tests/test_exdark_download.py` 3개(표준 폴더 이름, 다른 폴더 이름, 다시 실행) + `test_requirements_include_gdown` → 실패 확인 후 통과. 전체 82개 통과, 6개 skip

### 전체 실험 결과 (run 20261111-011401, BDD100K 야간 300장 + ExDark 300장, 9개 조건)
- 소요 시간 약 11분. 건너뛴 조건 0개. 제외 이미지 4장: 모두 ExDark의 큰 이미지에서 Zero-DCE++가 GPU 메모리 부족(OOM)으로 실패 → ExDark Zero-DCE++만 296장 기준 (다른 조건은 300장)
- BDD100K(정답 3,271개, car 2,852 / person 323) 전체 F1 / AP50: 원본 0.484 / 0.312, Gamma1.5 0.473 / 0.297, Gamma2.0 0.441 / 0.271, Gamma2.5 0.426 / 0.260, CLAHE2.0 0.490 / 0.311, CLAHE3.0 0.476 / 0.295, CLAHE4.0 0.455 / 0.261, Zero-DCE++ 0.373 / 0.199, Real-ESRGAN 0.486 / 0.287
- ExDark(정답 1,063개, person 587 / car 259) 전체 F1 / AP50: 원본 0.714 / 0.742, Gamma1.5 0.705 / 0.731, Gamma2.0 0.678 / 0.681, Gamma2.5 0.669 / 0.657, CLAHE2.0 0.713 / 0.731, CLAHE3.0 0.711 / 0.731, CLAHE4.0 0.709 / 0.727, Zero-DCE++ 0.668 / 0.689, Real-ESRGAN 0.703 / 0.728
- 화질 개선 시간(평균 ms/장, BDD / ExDark): Gamma 0.7 / 0.9, CLAHE 11 / 11, Zero-DCE++ 49 / 126, Real-ESRGAN 818 / 787. 탐지 시간 12~18ms
- 관찰: 두 데이터셋 모두 원본보다 확실히 좋아진 기법은 없음. 약한 처리(CLAHE2.0, Gamma1.5, Real-ESRGAN)는 원본과 비슷하고, 강하게 처리할수록(Gamma 2.0→2.5, CLAHE 3→4) 성능이 떨어지고 FP가 늘어남. Zero-DCE++가 가장 크게 떨어짐(BDD F1 −0.11, FP 316→464)
- BDD Recall(0.35)이 ExDark(0.74)보다 크게 낮음 → BDD는 먼 거리의 작은 차량이 많음
- 남은 문제: Zero-DCE++ OOM으로 ExDark 비교 이미지 수가 달라짐(296 vs 300) → "모든 조건 같은 이미지" 원칙 위반. 수정 필요
