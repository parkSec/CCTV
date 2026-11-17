"""Streamlit 화면이 쓰는 데이터 준비 함수 (Streamlit 없이 테스트할 수 있게 분리)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eval.analysis import delta_table

SUMMARY_COLS = ["dataset", "condition", "precision", "recall", "f1", "ap50", "ap50_95", "n_gt"]


def list_runs(root: Path) -> list[str]:
    """run_meta.json이 있는 결과 폴더 이름, 최신순."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = [d for d in root.iterdir() if (d / "run_meta.json").exists()]
    return [d.name for d in sorted(runs, key=lambda d: (d / "run_meta.json").stat().st_mtime, reverse=True)]


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()


def load_run(run_dir: Path) -> dict:
    """{"detection", "iqa", "timing", "delta", "correlation", "selection", "meta"}.
    analysis/가 아직 없으면 delta는 바로 계산하고 correlation·selection은 비워 둔다."""
    run_dir = Path(run_dir)
    a = run_dir / "analysis"
    det = _csv(run_dir / "detection_metrics.csv")
    delta = _csv(a / "delta.csv")
    if delta.empty and not det.empty:
        delta = delta_table(det)
    sel_path = a / "selection.json"
    meta_path = run_dir / "run_meta.json"
    return {"detection": det, "iqa": _csv(run_dir / "iqa_metrics.csv"), "timing": _csv(run_dir / "timing.csv"),
            "delta": delta, "correlation": _csv(a / "iqa_correlation.csv"),
            "selection": json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else None,
            "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}}


def summary_table(det: pd.DataFrame, detector: str, imgsz: int) -> pd.DataFrame:
    """한 탐지 설정의 class·size가 all인 행만, 보기 좋은 열 순서로."""
    m = (det["class"] == "all") & (det["size"] == "all") & (det["detector"] == detector) & (det["imgsz"] == imgsz)
    return det.loc[m, [c for c in SUMMARY_COLS if c in det.columns]].reset_index(drop=True)
