import json
import os

import pandas as pd
import pytest

from app.views import list_runs, load_run, summary_table


def _run(root, name, mtime):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame([
        {"dataset": "x", "condition": c, "detector": "yolo26s_bdd", "imgsz": s, "class": k, "size": z,
         "precision": 0.5, "recall": 0.5, "f1": 0.5, "ap50": ap, "ap50_95": ap / 2, "tp": 1, "fp": 1, "fn": 1, "n_gt": 2}
        for c, ap in [("original", 0.4), ("gamma_g1.5", 0.5)] for s in (640, 960)
        for k in ("all", "car") for z in ("all", "small")]).to_csv(d / "detection_metrics.csv", index=False)
    pd.DataFrame([{"dataset": "x", "condition": "original", "enh_total_ms_mean": 0.1,
                   "det_yolo26s_bdd_640_ms_mean": 10}]).to_csv(d / "timing.csv", index=False)
    pd.DataFrame([{"dataset": "x", "condition": "original", "niqe_mean": 5.0}]).to_csv(d / "iqa_metrics.csv", index=False)
    (d / "run_meta.json").write_text(json.dumps({"run_id": name}), encoding="utf-8")
    os.utime(d / "run_meta.json", (mtime, mtime))
    return d


def test_list_runs_newest_first(tmp_path):
    _run(tmp_path, "older", 1_000)
    _run(tmp_path, "newer", 2_000)
    (tmp_path / "not_a_run").mkdir()
    assert list_runs(tmp_path) == ["newer", "older"]


def test_list_runs_missing_root(tmp_path):
    assert list_runs(tmp_path / "none") == []


def test_load_run_computes_delta_without_analysis(tmp_path):
    r = load_run(_run(tmp_path, "r", 1_000))
    assert set(r) >= {"detection", "iqa", "timing", "delta", "correlation"}
    d = r["delta"]
    assert d.loc[(d["condition"] == "gamma_g1.5") & (d["imgsz"] == 640), "d_ap50"].iloc[0] == pytest.approx(0.1)


def test_summary_table_only_all_rows(tmp_path):
    det = load_run(_run(tmp_path, "r", 1_000))["detection"]
    t = summary_table(det, "yolo26s_bdd", 640)
    assert len(t) == 2 and list(t["condition"]) == ["original", "gamma_g1.5"]
    assert "class" not in t.columns and "ap50_95" in t.columns
