"""실험 결과 폴더를 분석해 <run_dir>/analysis/에 표와 그림을 만든다.

사용: python -m scripts.analyze results/<run_id>
- delta.csv: 원본 대비 지표 변화량 (탐지 설정별)
- iqa_correlation.csv: 화질 지표 평균과 ΔAP50의 스피어만 상관 (기준 설정 yolo26s_bdd 640)
- selection.json: 빠른 모델·확실한 모델 (기준 설정)
- selection_other_settings.csv: 나머지 탐지 설정에서 같은 규칙으로 고른 결과 (결론이 설정에 따라 바뀌는지)
- speed_vs_dmap.png, dmap_by_dataset.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.analysis import BUDGET_MS, REF_DETECTOR, REF_IMGSZ, delta_table, iqa_correlation, select_models

# 차트 색 (dataviz 기본 팔레트): 계열 1 파랑, 음수는 발산 쌍의 빨강
BLUE, RED, TEXT2, GRID, SURFACE = "#2a78d6", "#e34948", "#52514e", "#e6e5e1", "#fcfcfb"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):   # 한글 축 제목용
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def plot_speed_vs_dmap(table: pd.DataFrame, path: Path, budget_ms: float = BUDGET_MS) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, layout="constrained")
    ax.grid(color=GRID, linewidth=0.8)
    ax.axvline(budget_ms, color=TEXT2, linewidth=1, linestyle="--")
    ax.text(budget_ms, 0.98, f" {budget_ms}ms (30fps)", color=TEXT2, fontsize=8, va="top",
            transform=ax.get_xaxis_transform())   # y는 축 높이 기준 → 데이터 범위와 무관
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    ax.scatter(table["total_ms"], table["mean_d_ap50"], s=40, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for _, r in table.iterrows():   # 점마다 조건 이름 (색이 아니라 글자로 구분)
        ax.annotate(r["condition"], (r["total_ms"], r["mean_d_ap50"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=TEXT2)
    ax.set_xlabel("개선 + 탐지 시간 (ms, 데이터셋 평균)")
    ax.set_ylabel("ΔAP50 (원본 대비, 데이터셋 평균)")
    ax.set_title(f"처리 시간과 탐지 변화 ({REF_DETECTOR} {REF_IMGSZ})", loc="left", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def plot_dmap_by_dataset(delta: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = delta[(delta["detector"] == REF_DETECTOR) & (delta["imgsz"] == REF_IMGSZ) & (delta["condition"] != "original")]
    datasets = sorted(d["dataset"].unique()) or ["-"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets) + 1.2, 4.0), dpi=150, sharey=True,
                             squeeze=False, layout="constrained")
    conds = sorted(d["condition"].unique())
    for ax, ds in zip(axes[0], datasets):
        g = d[d["dataset"] == ds].set_index("condition").reindex(conds)
        vals = g["d_ap50"].fillna(0)
        ax.barh(conds, vals, color=[BLUE if v >= 0 else RED for v in vals], height=0.6, edgecolor=SURFACE,
                linewidth=2)
        ax.axvline(0, color=TEXT2, linewidth=0.8)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=TEXT2)
        ax.set_title(ds, loc="left", fontsize=9)
        lo, hi = min(vals.min(), 0), max(vals.max(), 0)
        pad = (hi - lo or 0.01) * 0.35   # 막대 끝 숫자가 들어갈 자리
        ax.set_xlim(lo - (pad if lo < 0 else pad * 0.15), hi + (pad if hi > 0 else pad * 0.15))
    axes[0][0].invert_yaxis()
    fig.suptitle(f"데이터셋별 ΔAP50 ({REF_DETECTOR} {REF_IMGSZ}, 파랑 = 좋아짐, 빨강 = 나빠짐)", x=0.01, ha="left",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def analyze(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    det = pd.read_csv(run_dir / "detection_metrics.csv")
    timing = pd.read_csv(run_dir / "timing.csv")
    iqa_path = run_dir / "iqa_metrics.csv"
    iqa = pd.read_csv(iqa_path) if iqa_path.exists() and iqa_path.stat().st_size > 5 else pd.DataFrame()
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    delta = delta_table(det)
    delta.to_csv(out / "delta.csv", index=False, encoding="utf-8-sig")
    if len(iqa):
        corr = iqa_correlation(iqa, delta, lower_better=meta.get("iqa", {}).get("lower_better"))
        corr.to_csv(out / "iqa_correlation.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["dataset", "metric", "spearman", "n_conditions", "lower_better"]).to_csv(
            out / "iqa_correlation.csv", index=False, encoding="utf-8-sig")

    sel = select_models(delta, timing)
    sel["table"].to_csv(out / "selection_table.csv", index=False, encoding="utf-8-sig")
    (out / "selection.json").write_text(json.dumps(
        {"detector": REF_DETECTOR, "imgsz": REF_IMGSZ, "budget_ms": BUDGET_MS, "fast": sel["fast"],
         "reliable": sel["reliable"], "table": sel["table"].to_dict(orient="records")},
        ensure_ascii=False, indent=2), encoding="utf-8")

    others = []
    for (name, size), _ in det.groupby(["detector", "imgsz"]):
        if (name, size) == (REF_DETECTOR, REF_IMGSZ) or f"det_{name}_{size}_ms_mean" not in timing.columns:
            continue
        s = select_models(delta, timing, detector=name, imgsz=int(size))
        others.append({"detector": name, "imgsz": int(size), "fast": s["fast"], "reliable": s["reliable"]})
    pd.DataFrame(others, columns=["detector", "imgsz", "fast", "reliable"]).to_csv(
        out / "selection_other_settings.csv", index=False, encoding="utf-8-sig")

    plot_speed_vs_dmap(sel["table"], out / "speed_vs_dmap.png")
    plot_dmap_by_dataset(delta, out / "dmap_by_dataset.png")
    print(f"빠른 모델: {sel['fast']}  /  확실한 모델: {sel['reliable']}  ({REF_DETECTOR} {REF_IMGSZ})")
    print(f"분석 결과: {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    analyze(Path(ap.parse_args(argv).run_dir))


if __name__ == "__main__":
    main()
