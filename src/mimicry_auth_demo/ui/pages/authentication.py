"""Authentication page."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from ...config import AppConfig
from ...core.authentication import AuthenticationSession
from ...storage.profile_store import ProfileStore
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output
from ..components.stability_chart import render_stability_chart


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Аутентификация")

    profiles = store.list_profiles()
    if not profiles:
        st.warning("Нет зарегистрированных профилей. Перейдите в **Регистрация**.")
        return

    user_id = st.selectbox("Выберите профиль", profiles)
    container = store.load_profile(user_id)

    capture_mode = st.radio("Источник", ["Веб-камера", "Файл"], horizontal=True)

    result = None

    if capture_mode == "Файл":
        uploaded = st.file_uploader("Загрузить видео", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Аутентифицировать"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Обработка..."):
                session = AuthenticationSession(container, config)
                result = session.authenticate(tmp_path)
            os.unlink(tmp_path)
    else:
        frames_key = "auth_cam_frames"
        img = st.camera_input("Снять кадр")
        if img is not None:
            file_bytes = np.frombuffer(img.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                buf = st.session_state.get(frames_key, [])
                buf.append(frame_bgr)
                st.session_state[frames_key] = buf
                st.info(f"Кадров: {len(buf)}")

        if st.session_state.get(frames_key) and st.button("Аутентифицировать"):
            frames = st.session_state.pop(frames_key)
            with st.spinner("Обработка..."):
                session = AuthenticationSession(container, config)
                result = session.authenticate_frames(frames)

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

    # Timings
    if result.pipeline_timings_ms:
        st.markdown("**Время выполнения стадий (мс):**")
        timing_cols = st.columns(len(result.pipeline_timings_ms))
        for col, (stage, ms) in zip(timing_cols, result.pipeline_timings_ms.items()):
            col.metric(stage, f"{ms:.0f}")

    # Pipeline inspector
    st.markdown("---")
    st.subheader("Pipeline Inspector")
    selected_stage = st.selectbox("Стадия", STAGE_NAMES, key="auth_stage_sel")
    render_stage_output(selected_stage, result.stage_output)
