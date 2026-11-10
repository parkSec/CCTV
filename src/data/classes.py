"""공통 클래스 정의와 데이터셋별 매핑표. 클래스 매핑은 이 파일에서만 관리한다."""

COMMON_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# YOLO(COCO) 클래스 id → 공통 클래스 id. 여기에 없는 COCO 클래스는 버린다.
COCO_TO_COMMON = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5}

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
