import json

import numpy as np
import pandas as pd
import pytest

from src.eval.analysis import delta_table, iqa_correlation, select_models


def _det(rows):
    """rows: (dataset, condition, ap50) → detection_metrics.csv 형식 (yolo26s_bdd 640, class/size all)."""
    out = []
    for ds, cond, ap in rows:
        out.append({"dataset": ds, "condition": cond, "detector": "yolo26s_bdd", "imgsz": 640,
                    "class": "all", "size": "all", "precision": ap, "recall": ap, "f1": ap,
                    "ap50": ap, "ap50_95": ap / 2})
        out.append({"dataset": ds, "condition": cond, "detector": "yolo26s_bdd", "imgsz": 640,
                    "class": "car", "size": "all", "precision": 0, "recall": 0, "f1": 0, "ap50": 0, "ap50_95": 0})
    return pd.DataFrame(out)


def _timing(rows, det_ms=10.0):
    return pd.DataFrame([{"dataset": ds, "condition": c, "enh_total_ms_mean": ms,
                          "det_yolo26s_bdd_640_ms_mean": det_ms} for ds, c, ms in rows])


def test_delta_original_zero():
    d = delta_table(_det([("x", "original", 0.5), ("x", "A", 0.6)]))
    assert d.loc[d["condition"] == "original", "d_ap50"].iloc[0] == 0
    assert d.loc[d["condition"] == "A", "d_ap50"].iloc[0] == pytest.approx(0.1)
    assert d.loc[d["condition"] == "A", "d_ap50_95"].iloc[0] == pytest.approx(0.05)
    assert len(d) == 2   # class·size가 all인 행만


def test_delta_per_setting_and_dataset():
    det = pd.concat([_det([("x", "original", 0.5), ("x", "A", 0.6)]),
                     _det([("y", "original", 0.2), ("y", "A", 0.1)])])
    d = delta_table(det)
    assert d.set_index(["dataset", "condition"]).loc[("y", "A"), "d_ap50"] == pytest.approx(-0.1)


def test_select_fast_respects_budget():
    delta = delta_table(_det([("x", "original", 0.5), ("x", "A", 0.52), ("x", "B", 0.55)]))
    t = _timing([("x", "original", 0.1), ("x", "A", 10.0), ("x", "B", 40.0)])
    s = select_models(delta, t)
    assert s["fast"] == "A" and s["reliable"] == "B"
    assert set(s["table"]["condition"]) == {"A", "B"}


def test_select_fast_none_when_all_slow():
    delta = delta_table(_det([("x", "original", 0.5), ("x", "A", 0.52)]))
    s = select_models(delta, _timing([("x", "original", 0.1), ("x", "A", 30.0)]))
    assert s["fast"] is None and s["reliable"] == "A"


def test_reliable_uses_rank_mean():
    # x: A 1위, B 2위, C 3위 / y: C 1위, B 2위, A 3위 → 순위 평균 모두 2.0, 평균 d_ap50은 C가 가장 큼
    det = pd.concat([_det([("x", "original", 0.5), ("x", "A", 0.55), ("x", "B", 0.53), ("x", "C", 0.51)]),
                     _det([("y", "original", 0.5), ("y", "A", 0.50), ("y", "B", 0.52), ("y", "C", 0.60)])])
    t = _timing([(ds, c, 1.0) for ds in "xy" for c in ["original", "A", "B", "C"]])
    s = select_models(delta_table(det), t)
    assert s["table"].set_index("condition")["rank_mean"].tolist() == [2.0, 2.0, 2.0]
    assert s["reliable"] == "C"


def test_spearman_perfect():
    delta = delta_table(_det([("x", "original", 0.5), ("x", "A", 0.6), ("x", "B", 0.7)]))
    iqa = pd.DataFrame({"dataset": "x", "condition": ["original", "A", "B"],
                        "niqe_mean": [9.0, 8.0, 7.0], "musiq_mean": [10.0, 20.0, 30.0]})
    c = iqa_correlation(iqa, delta, lower_better={"niqe": True, "musiq": False}).set_index("metric")
    assert c.loc["musiq", "spearman"] == pytest.approx(1.0)
    assert c.loc["niqe", "spearman"] == pytest.approx(-1.0)
    assert bool(c.loc["niqe", "lower_better"]) and c.loc["musiq", "n_conditions"] == 3


def test_spearman_skips_empty_metric():
    delta = delta_table(_det([("x", "original", 0.5), ("x", "A", 0.6)]))
    iqa = pd.DataFrame({"dataset": "x", "condition": ["original", "A"], "psnr_mean": [np.nan, np.nan]})
    c = iqa_correlation(iqa, delta)
    assert c.iloc[0]["n_conditions"] == 0 and np.isnan(c.iloc[0]["spearman"])


def test_analyze_script_writes_outputs(tmp_path):
    from scripts.analyze import analyze
    det = pd.concat([_det([("x", "original", 0.5), ("x", "A", 0.55), ("x", "B", 0.6)])])
    det.to_csv(tmp_path / "detection_metrics.csv", index=False)
    _timing([("x", "original", 0.1), ("x", "A", 5.0), ("x", "B", 50.0)]).to_csv(tmp_path / "timing.csv", index=False)
    pd.DataFrame({"dataset": "x", "condition": ["original", "A", "B"],
                  "niqe_mean": [9.0, 8.0, 7.0]}).to_csv(tmp_path / "iqa_metrics.csv", index=False)
    (tmp_path / "run_meta.json").write_text(json.dumps({"iqa": {"lower_better": {"niqe": True}}}), encoding="utf-8")
    out = analyze(tmp_path)
    for name in ["delta.csv", "iqa_correlation.csv", "selection.json", "speed_vs_dmap.png",
                 "dmap_by_dataset.png", "selection_other_settings.csv"]:
        assert (out / name).exists(), name
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert sel["fast"] == "A" and sel["reliable"] == "B"
