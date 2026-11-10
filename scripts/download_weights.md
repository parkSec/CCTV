# 가중치 받기

모두 프로젝트의 `weights\` 폴더에 넣습니다. 파일 이름은 `configs/default.yaml`과 같아야 합니다.

| 파일 | 용도 | 받는 곳 |
|---|---|---|
| `yolo26s.pt` | 탐지기 | 따로 받을 필요 없음. ultralytics 8.4.0 이상이 처음 실행할 때 자동으로 받음 |
| `Epoch99.pth` | Zero-DCE++ | 공식 저장소 https://github.com/Li-Chongyi/Zero-DCE_extension 의 `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth` (저장소 README 기준 위치) |
| `RealESRGAN_x2plus.pth` | Real-ESRGAN | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth |

## 확인 방법

```bash
pytest tests/test_zerodce.py tests/test_realesrgan.py -rs
```
- `skipped`가 남아 있으면 그 이유(가중치 없음 / 패키지 없음)가 함께 출력됩니다.
- Zero-DCE++ 가중치가 구조와 맞지 않으면 실험 시 `run_meta.json`의 `skipped_conditions`에 "가중치 구조가 Zero-DCE++와 맞지 않음"과 맞지 않는 키 목록이 남습니다. 그 내용을 알려 주세요.

## 참고
- Zero-DCE / Zero-DCE++ 코드는 비상업적 연구 목적으로만 사용할 수 있습니다 (수업 프로젝트는 해당).
- Zero-DCE++ 논문의 기본 축소 배율은 12입니다. 이 프로젝트는 스펙에 따라 1을 씁니다(`configs/default.yaml`의 `scale_factor`).
