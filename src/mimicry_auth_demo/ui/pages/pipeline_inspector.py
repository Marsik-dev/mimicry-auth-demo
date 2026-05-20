"""
Pipeline Inspector page — live webcam feed with per-stage visualization
and real-time parameter editing.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from mimicry_preproc import Pipeline

from ...config import AppConfig
from ..components.pipeline_settings import (
    export_config_button,
    get_current_config,
    render_stage_settings,
)
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output


def render(config: AppConfig) -> None:
    st.title("Pipeline Inspector")
    st.markdown("Настраивай параметры каждой стадии в реальном времени и смотри выход.")

    # Sidebar stage selector
    with st.sidebar:
        st.markdown("### Стадии пайплайна")
        selected_stage = st.radio("Активная стадия", STAGE_NAMES, index=4, key="inspector_stage")
        st.markdown("---")
        st.markdown("### Параметры стадии")
        changed = render_stage_settings(selected_stage)
        st.markdown("---")
        current_cfg = get_current_config()
        export_config_button(current_cfg)

    capture_mode = st.radio("Источник", ["Веб-камера (кадр)", "Файл"], horizontal=True)

    stage_out = None

    if capture_mode == "Файл":
        uploaded = st.file_uploader("Видео файл", type=["mp4", "avi", "mov"])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            if st.button("Запустить", type="primary") or changed:
                pipeline = Pipeline(current_cfg)
                with st.spinner("Обработка..."):
                    stage_out = pipeline.run_with_debug(tmp_path)
                st.session_state["inspector_last_out"] = stage_out
            os.unlink(tmp_path)
    else:
        img = st.camera_input("Снять кадр")
        frames_key = "inspector_frames"
        if img is not None:
            file_bytes = np.frombuffer(img.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                buf = st.session_state.get(frames_key, [])
                buf.append(frame_bgr)
                st.session_state[frames_key] = buf

        col1, col2 = st.columns(2)
        if col1.button("▶ Запустить пайплайн", type="primary"):
            frames = st.session_state.get(frames_key, [])
            if frames:
                pipeline = Pipeline(current_cfg)
                with st.spinner("Обработка..."):
                    fv = pipeline.run_from_frames(frames)
                    from mimicry_preproc.pipeline import StageOutput
                    stage_out = StageOutput(feature_vector=fv)
                st.session_state["inspector_last_out"] = stage_out
        if col2.button("🗑 Очистить буфер"):
            st.session_state.pop(frames_key, None)
            st.rerun()

        n = len(st.session_state.get(frames_key, []))
        if n:
            st.caption(f"В буфере: {n} кадров")

    if changed:
        st.rerun()

    # Always show last result
    last_out = stage_out or st.session_state.get("inspector_last_out")
    if last_out is not None:
        st.markdown("---")
        render_stage_output(selected_stage, last_out)
    else:
        st.info("Запусти пайплайн чтобы увидеть вывод стадии.")
