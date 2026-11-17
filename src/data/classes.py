"""공통 클래스 정의와 데이터셋별 매핑표. 클래스 매핑은 이 파일에서만 관리한다."""

COMMON_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# BDD100K로 학습한 탐지 모델(dronefreak/bdd100k-yolo26s·m)의 클래스 이름 → 공통 클래스 id.
# 모델 카드 순서: person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign
BDD_MODEL_TO_COMMON = {"person": 0, "rider": 0, "bike": 1, "bicycle": 1, "car": 2,
                       "motor": 3, "motorcycle": 3, "bus": 4, "truck": 5}
# 공통 클래스에 없어 버리는 모델 클래스
BDD_MODEL_IGNORED = {"train", "traffic light", "traffic sign"}

# BDD100K: det_20 라벨 이름과 2018 원본 라벨 이름을 모두 받는다.
BDD_TO_COMMON = {
    "pedestrian": 0, "person": 0, "rider": 0,
    "bicycle": 1, "bike": 1,
    "car": 2,
    "motorcycle": 3, "motor": 3,
    "bus": 4,
    "truck": 5,
}

# ExDark: 12개 클래스 중 도로 관련 5개만 사용 (truck 없음)
EXDARK_TO_COMMON = {"People": 0, "Bicycle": 1, "Car": 2, "Motorbike": 3, "Bus": 4}

# ACDC: COCO 형식 박스 라벨(gt_detection)의 카테고리 이름. train(열차)은 공통 클래스에 없어 버린다.
ACDC_TO_COMMON = {"person": 0, "rider": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 4, "truck": 5}

# DarkDriving: labelme 라벨은 car 하나 (실제 night val 3,564개 전부 car)
DARKDRIVING_TO_COMMON = {"car": 2, "Car": 2}
