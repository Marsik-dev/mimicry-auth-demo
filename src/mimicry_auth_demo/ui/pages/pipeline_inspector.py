"""
Pipeline Inspector — live webcam feed, per-stage visualization,
real-time parameter editing with persistent session state.
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from mimicry_preproc import Pipeline

from ...config import AppConfig
from ..components.pipeline_settings import (
    export_config_button,
    get_current_config,
    render_stage_settings,
)
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output
from ..components.webcam_capture import webcam_recorder


def render(config: AppConfig) -> None:
    st.title("Pipeline Inspector")
    st.markdown(
        "Настраивай параметры каждой стадии и смотри её вывод в реальном времени."
    )

    with st.sidebar:
        st.markdown("### Стадии пайплайна")
        selected_stage = st.radio(
            "Активная стадия", STAGE_NAMES, index=4, key="inspector_stage"
        )
        st.markdown("---")
        st.markdown("### Параметры стадии")
        changed = render_stage_settings(selected_stage)
        st.markdown("---")
        current_cfg = get_current_config()
        export_config_button(current_cfg)

    tab_cam, tab_file = st.tabs(["📷 Веб-камера", "📁 Файл"])

    stage_out = None

    with tab_cam:
        st.markdown("Запиши видео, затем нажми **Запустить** чтобы увидеть вывод выбранной стадии.")
        frames = webcam_recorder(key="inspector_webcam", label="камера инспектора")
        if frames is not None:
            pipeline = Pipeline(current_cfg)
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                stage_out = pipeline.run_with_debug_from_frames(frames)
            st.session_state["inspector_last_out"] = stage_out

    with tab_file:
        uploaded = st.file_uploader("Видео", type=["mp4", "avi", "mov"])
        if uploaded:
            if st.button("▶ Запустить пайплайн", type="primary") or changed:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                pipeline = Pipeline(current_cfg)
                with st.spinner("Обработка..."):
                    stage_out = pipeline.run_with_debug(tmp_path)
                os.unlink(tmp_path)
                st.session_state["inspector_last_out"] = stage_out

    if changed:
        st.rerun()

    last_out = stage_out or st.session_state.get("inspector_last_out")
    if last_out is not None:
        st.markdown("---")
        render_stage_output(selected_stage, last_out)
    else:
        st.info("Запусти пайплайн чтобы увидеть вывод стадии.")
