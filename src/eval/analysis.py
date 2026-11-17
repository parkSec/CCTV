"""실험 결과 분석: 원본 대비 변화량, 화질 지표와 탐지 변화의 상관, 빠른 모델·확실한 모델 선택 (스펙 7장)."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["precision", "recall", "f1", "ap50", "ap50_95"]
KEYS = ["dataset", "detector", "imgsz"]
REF_DETECTOR, REF_IMGSZ = "yolo26s_bdd", 640   # 선택 기준 탐지 설정
BUDGET_MS = 33.3                                # 실시간 30fps


def delta_table(det: pd.DataFrame) -> pd.DataFrame:
    """class·size가 all인 행에서 (dataset, detector, imgsz)마다 각 조건 값 − original 값 → d_<지표>."""
    a = det[(det["class"] == "all") & (det["size"] == "all")].copy()
    base = a[a["condition"] == "original"].set_index(KEYS)[METRICS]
    a = a.join(base, on=KEYS, rsuffix="_orig")
    for m in METRICS:
        a[f"d_{m}"] = a[m] - a[f"{m}_orig"]
    return a[KEYS + ["condition"] + METRICS + [f"d_{m}" for m in METRICS]].reset_index(drop=True)


def _setting(df: pd.DataFrame, detector: str, imgsz: int) -> pd.DataFrame:
    return df[(df["detector"] == detector) & (df["imgsz"] == imgsz)]


def iqa_correlation(iqa: pd.DataFrame, delta: pd.DataFrame, detector: str = REF_DETECTOR, imgsz: int = REF_IMGSZ,
                    lower_better: dict[str, bool] | None = None) -> pd.DataFrame:
    """데이터셋 × 화질 지표마다, 조건별 <지표>_mean과 d_ap50의 스피어만 상관계수.
    지표 방향은 바꾸지 않는다(lower_better 열로 함께 남김: True면 음의 상관이 '화질 좋을수록 탐지 좋음')."""
    lower_better = lower_better or {}
    d = _setting(delta, detector, imgsz)[["dataset", "condition", "d_ap50"]]
    metrics = [c.removesuffix("_mean") for c in iqa.columns if c.endswith("_mean")]
    rows = []
    for ds, g in iqa.merge(d, on=["dataset", "condition"]).groupby("dataset"):
        for m in metrics:
            pair = g[[f"{m}_mean", "d_ap50"]].astype(float).dropna()
            rho = pair[f"{m}_mean"].corr(pair["d_ap50"], method="spearman") if len(pair) >= 2 else np.nan
            rows.append({"dataset": ds, "metric": m, "spearman": rho, "n_conditions": len(pair),
                         "lower_better": lower_better.get(m)})
    return pd.DataFrame(rows, columns=["dataset", "metric", "spearman", "n_conditions", "lower_better"])


def select_models(delta: pd.DataFrame, timing: pd.DataFrame, budget_ms: float = BUDGET_MS,
                  detector: str = REF_DETECTOR, imgsz: int = REF_IMGSZ) -> dict:
    """{"fast": 조건 또는 None, "reliable": 조건, "table": DataFrame}.
    - 합계 시간 = 개선 total 평균 + 탐지 평균, 데이터셋 평균
    - fast: 합계 ≤ budget_ms 중 데이터셋 평균 d_ap50 최대
    - reliable: 데이터셋마다 d_ap50 순위(1 = 최고)의 평균이 가장 작은 조건, 동점이면 평균 d_ap50이 큰 쪽"""
    d = _setting(delta, detector, imgsz)
    d = d[d["condition"] != "original"][["dataset", "condition", "d_ap50"]].copy()
    d["rank"] = d.groupby("dataset")["d_ap50"].rank(ascending=False, method="average")
    t = timing[timing["condition"] != "original"].copy()
    t["total_ms"] = t["enh_total_ms_mean"] + t[f"det_{detector}_{imgsz}_ms_mean"]
    table = (d.groupby("condition").agg(mean_d_ap50=("d_ap50", "mean"), rank_mean=("rank", "mean"),
                                        n_datasets=("dataset", "nunique"))
             .join(t.groupby("condition")["total_ms"].mean())
             .reset_index())
    table["within_budget"] = table["total_ms"] <= budget_ms
    fast = table[table["within_budget"]].sort_values("mean_d_ap50", ascending=False)
    reliable = table.sort_values(["rank_mean", "mean_d_ap50"], ascending=[True, False])
    return {"fast": fast["condition"].iloc[0] if len(fast) else None,
            "reliable": reliable["condition"].iloc[0] if len(reliable) else None,
            "table": table.sort_values("condition").reset_index(drop=True)}
