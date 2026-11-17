"""실험 결과 화면. 실행: streamlit run app/streamlit_app.py

탭 3개: (1) 결과 — 지표표, ΔAP50, 속도-ΔAP50, 빠른/확실한 모델, 화질 지표와 상관
        (2) 비교 이미지 — 실험이 저장한 samples/
        (3) 데모 — 이미지 또는 영상(+프레임 번호)을 올리면 조건 8개 비교 격자
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app.views import list_runs, load_run, summary_table  # noqa: E402
from src.config import load_config  # noqa: E402

st.set_page_config(page_title="야간 화질 개선 × 탐지 비교", layout="wide")
cfg = load_config(ROOT / "configs" / "default.yaml")
results_root = Path(cfg["output"]["root"])


@st.cache_resource
def _enhancers():
    from src.enhance.registry import build_enhancers
    return build_enhancers(cfg["conditions"], cfg["third_party"]["root"])


@st.cache_resource
def _detector(name: str, imgsz: int):
    from src.detect.yolo import YoloDetector
    d = next(d for d in cfg["detectors"] if d["name"] == name)
    return YoloDetector(d["weights"], imgsz=imgsz, conf=cfg["detection"]["conf_min"],
                        device=cfg["detection"].get("device", 0), name=name)


tab_res, tab_samples, tab_demo = st.tabs(["결과", "비교 이미지", "데모"])

runs = list_runs(results_root)
with tab_res:
    if not runs:
        st.info("결과가 없습니다. `python -m scripts.run_experiment`로 실험을 먼저 돌리세요.")
    else:
        run = st.selectbox("실험", runs)
        r = load_run(results_root / run)
        det = r["detection"]
        settings = sorted({(d, int(s)) for d, s in zip(det["detector"], det["imgsz"])}) if len(det) else []
        if settings:
            c1, c2 = st.columns(2)
            name = c1.selectbox("탐지 모델", sorted({s[0] for s in settings}))
            imgsz = c2.selectbox("입력 크기", sorted({s[1] for s in settings if s[0] == name}))
            st.subheader("탐지 지표 (클래스·크기 전체)")
            st.dataframe(summary_table(det, name, imgsz), use_container_width=True)
            d = r["delta"]
            d = d[(d["detector"] == name) & (d["imgsz"] == imgsz) & (d["condition"] != "original")]
            st.subheader("원본 대비 ΔAP50")
            st.bar_chart(d.pivot_table(index="condition", columns="dataset", values="d_ap50"))
        if r["selection"]:
            s = r["selection"]
            st.subheader(f"선택 결과 ({s['detector']} {s['imgsz']}, 실시간 기준 {s['budget_ms']}ms)")
            c1, c2 = st.columns(2)
            c1.metric("빠른 모델", s["fast"] or "기준 안에 드는 모델 없음")
            c2.metric("확실한 모델", s["reliable"] or "-")
            st.dataframe(s["table"], use_container_width=True)
            for png in ("speed_vs_dmap.png", "dmap_by_dataset.png"):
                p = results_root / run / "analysis" / png
                if p.exists():
                    st.image(str(p))
        else:
            st.caption(f"분석 결과가 없습니다: `python -m scripts.analyze {results_root.name}/{run}`")
        st.subheader("처리 시간")
        st.dataframe(r["timing"], use_container_width=True)
        if len(r["iqa"]):
            st.subheader("화질 지표")
            st.dataframe(r["iqa"], use_container_width=True)
        if len(r["correlation"]):
            st.subheader("화질 지표와 ΔAP50의 스피어만 상관")
            st.dataframe(r["correlation"], use_container_width=True)

with tab_samples:
    if runs:
        run = st.selectbox("실험 ", runs)
        sdir = results_root / run / "samples"
        datasets = sorted(p.name for p in sdir.iterdir()) if sdir.exists() else []
        if datasets:
            ds = st.selectbox("데이터셋", datasets)
            conds = sorted(p.name for p in (sdir / ds).iterdir())
            images = sorted(p.name for p in (sdir / ds / conds[0]).glob("*.jpg"))
            if images:
                img_name = st.selectbox("이미지", images)
                cols = st.columns(4)
                for i, c in enumerate(conds):
                    p = sdir / ds / c / img_name
                    if p.exists():
                        cols[i % 4].image(str(p), caption=c)
        else:
            st.info("저장된 비교 이미지가 없습니다.")

with tab_demo:
    from src.data.resize import resize_max_side
    from src.frames import extract_frame
    from src.imageio import read_image
    from src.visualize import compare_frame

    up = st.file_uploader("이미지 또는 영상", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("탐지 모델 ", [d["name"] for d in cfg["detectors"]])
    imgsz = c2.selectbox("입력 크기 ", cfg["detection"]["imgsz"])
    frame = c3.number_input("프레임 번호 (영상)", min_value=0, value=0, step=1)
    if up is not None and st.button("비교"):
        suffix = Path(up.name).suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"input{suffix}"
            p.write_bytes(up.getvalue())
            img = extract_frame(p, frame_index=int(frame)) if suffix in {".mp4", ".avi", ".mov", ".mkv"} else read_image(p)
        if img is None:
            st.error("이미지를 읽을 수 없습니다.")
        else:
            img, _ = resize_max_side(img, cfg["preprocess"]["max_side"])
            enhancers, skipped = _enhancers()
            with st.spinner("개선·탐지 중"):
                grid, rows = compare_frame(img, enhancers, _detector(name, int(imgsz)), cfg["detection"]["conf_eval"])
            st.image(grid[:, :, ::-1], use_container_width=True)
            st.dataframe(rows, use_container_width=True)
            for s in skipped:
                st.caption(f"건너뜀 {s['condition']}: {s['reason']}")
