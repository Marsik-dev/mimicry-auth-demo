"""
Pipeline Inspector — live webcam feed, per-stage visualization,
real-time parameter editing with persistent session state.

Tabs:
  🔴 Live        — real-time processing overlay on webcam stream
  📷 Webcam      — record, then run full pipeline and inspect result
  📁 File        — upload video, run pipeline, inspect result
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
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.webcam_capture import webcam_recorder


def render(config: AppConfig) -> None:
    st.title("Pipeline Inspector")

    with st.sidebar:
        st.markdown("### Параметры стадии")
        # Stage selector only relevant for non-live tabs
        selected_stage = st.radio(
            "Стадия (для вкладок Webcam/File)",
            STAGE_NAMES, index=4, key="inspector_stage",
        )
        st.markdown("---")
        changed = render_stage_settings(selected_stage)
        st.markdown("---")
        export_config_button(get_current_config())

    tab_live, tab_cam, tab_file = st.tabs(["🔴 Live", "📷 Webcam", "📁 File"])

    # ------------------------------------------------------------------
    # Tab: Live real-time processing
    # ------------------------------------------------------------------
    with tab_live:
        st.markdown(
            "Камера обрабатывается **кадр за кадром** в реальном времени. "
            "Выбери стадию — её вывод рисуется поверх видео."
        )
        realtime_pipeline_widget(key="inspector_live", initial_mode="landmark_extractor")

    # ------------------------------------------------------------------
    # Tab: record then process
    # ------------------------------------------------------------------
    with tab_cam:
        st.markdown("Запиши видео, затем запусти полный пайплайн и изучи вывод любой стадии.")
        frames = webcam_recorder(key="inspector_webcam", label="камера инспектора")
        if frames is not None:
            pipeline = Pipeline(get_current_config())
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                stage_out = pipeline.run_with_debug_from_frames(frames)
            st.session_state["inspector_last_out"] = stage_out

    # ------------------------------------------------------------------
    # Tab: file
    # ------------------------------------------------------------------
    with tab_file:
        uploaded = st.file_uploader("Видео", type=["mp4", "avi", "mov"])
        if uploaded:
            if st.button("▶ Запустить пайплайн", type="primary") or changed:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                pipeline = Pipeline(get_current_config())
                with st.spinner("Обработка..."):
                    stage_out = pipeline.run_with_debug(tmp_path)
                os.unlink(tmp_path)
                st.session_state["inspector_last_out"] = stage_out

    if changed:
        st.rerun()

    # Show last result (for webcam/file tabs)
    last_out = st.session_state.get("inspector_last_out")
    if last_out is not None:
        st.markdown("---")
        st.subheader(f"Вывод стадии: {selected_stage}")
        render_stage_output(selected_stage, last_out)
