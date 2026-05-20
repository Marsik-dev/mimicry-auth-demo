"""Authentication page."""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from ...config import AppConfig
from ...storage.profile_store import ProfileStore
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.stability_chart import render_stability_chart
from ..components.webcam_capture import webcam_recorder


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Аутентификация")

    profiles = store.list_profiles()
    if not profiles:
        st.warning("Нет зарегистрированных профилей. Перейдите в **Регистрация**.")
        return

    user_id = st.selectbox("Профиль для проверки", profiles)
    container = store.load_profile(user_id)

    tab_live, tab_cam, tab_file = st.tabs(["🔴 Live preview", "📷 Запись", "📁 Файл"])

    result = None

    with tab_live:
        st.markdown("Убедись что лицо детектируется, прежде чем записывать.")
        realtime_pipeline_widget(key="auth_live", initial_mode="face_detector")

    with tab_cam:
        frames = webcam_recorder(key="auth_webcam", label="камера для аутентификации")
        if frames is not None:
            from ...core.authentication import AuthenticationSession
            session = AuthenticationSession(container, config)
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                result = session.authenticate_frames(frames)

    with tab_file:
        uploaded = st.file_uploader("Видео файл", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Аутентифицировать"):
            from ...core.authentication import AuthenticationSession
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            session = AuthenticationSession(container, config)
            with st.spinner("Обработка..."):
                result = session.authenticate(tmp_path)
            os.unlink(tmp_path)

    if result is not None:
        _show_result(result, container)


def _show_result(result, container) -> None:
    st.markdown("---")
    if result.accepted:
        st.success("✅ ДОСТУП РАЗРЕШЁН")
    else:
        st.error("❌ ДОСТУП ЗАПРЕЩЁН")

    col1, col2, col3 = st.columns(3)
    col1.metric("Расстояние Хэмминга", result.auth_result.hamming_distance)
    col2.metric("Порог", result.auth_result.details.get("hamming_threshold", "—"))
    col3.metric("Уверенность", f"{result.auth_result.confidence:.1%}")

    if result.pipeline_timings_ms:
        with st.expander("⏱ Время выполнения стадий (мс)"):
            cols = st.columns(len(result.pipeline_timings_ms))
            for col, (stage, ms) in zip(cols, result.pipeline_timings_ms.items()):
                col.metric(stage, f"{ms:.0f}")

    with st.expander("Pipeline Inspector"):
        selected_stage = st.selectbox("Стадия", STAGE_NAMES, key="auth_stage_sel")
        render_stage_output(selected_stage, result.stage_output)
