"""Enrollment page: record facial expression videos, train NPBK, save profile."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from ...config import AppConfig
from ...core.enrollment import EnrollmentSession
from ...storage.profile_store import ProfileStore
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output


_OTHER_VECTORS_PATH = os.path.join(os.path.dirname(__file__), "../../../../assets/other_vectors.npz")


def _load_other_vectors() -> np.ndarray | None:
    try:
        data = np.load(_OTHER_VECTORS_PATH)
        return data["vectors"]
    except Exception:
        return None


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Регистрация пользователя")

    user_id = st.text_input("Имя пользователя", value="user1", key="enroll_user_id")

    if "enroll_session" not in st.session_state or st.session_state.get("enroll_uid") != user_id:
        st.session_state["enroll_session"] = EnrollmentSession(config, user_id)
        st.session_state["enroll_uid"] = user_id
        st.session_state["enroll_last_stage_out"] = None

    session: EnrollmentSession = st.session_state["enroll_session"]
    progress = session.get_progress()

    st.progress(progress.progress_pct, text=f"Образцов: {progress.samples_collected} / {progress.samples_required}")

    if progress.sample_quality_scores:
        cols = st.columns(len(progress.sample_quality_scores))
        for col, score in zip(cols, progress.sample_quality_scores):
            col.metric("Качество", f"{score:.3f}")

    st.markdown("---")
    st.subheader("Запись образца")
    st.markdown("Запишите видео с одной эмоцией (улыбка, удивление, ...) — 4–5 секунд.")

    capture_mode = st.radio("Источник", ["Веб-камера", "Файл"], horizontal=True)

    if capture_mode == "Файл":
        uploaded = st.file_uploader("Загрузить видео", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Обработать"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Обработка..."):
                result = session.process_video(tmp_path)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            os.unlink(tmp_path)
            st.success(f"Добавлен образец. Качество: {result.quality_score:.3f}")
            st.rerun()

    else:
        frames_key = "enroll_cam_frames"
        img = st.camera_input("Снять кадр (или запишите видео)")
        if img is not None:
            file_bytes = np.frombuffer(img.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                buf = st.session_state.get(frames_key, [])
                buf.append(frame_bgr)
                st.session_state[frames_key] = buf
                st.info(f"Кадров в буфере: {len(buf)}")

        if st.session_state.get(frames_key) and st.button("Обработать буфер как образец"):
            frames = st.session_state.pop(frames_key)
            with st.spinner("Обработка..."):
                result = session.process_frames(frames)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            st.success(f"Добавлен образец. Качество: {result.quality_score:.3f}")
            st.rerun()

    # --- Pipeline inspector ---
    last_out = st.session_state.get("enroll_last_stage_out")
    if last_out is not None:
        st.markdown("---")
        st.subheader("Pipeline Inspector")
        selected_stage = st.selectbox("Выберите стадию для просмотра", STAGE_NAMES, key="enroll_stage_sel")
        render_stage_output(selected_stage, last_out)

    # --- Train ---
    st.markdown("---")
    if progress.is_ready:
        st.success("Достаточно образцов! Можно обучить НПБК.")
        if st.button("Обучить и сохранить профиль", type="primary"):
            other_vecs = _load_other_vectors()
            if other_vecs is None:
                st.error("Не найдены образцы 'Other' (assets/other_vectors.npz). "
                         "Добавьте файл с разнообразными образцами лиц других людей (≥64).")
                return
            with st.spinner("Обучение НПБК..."):
                training_result, container = session.train(other_vecs)
            store.save_profile(user_id, container)
            st.success(f"Профиль '{user_id}' сохранён!")
            st.metric("Средняя стабильность", f"{training_result.quality.mean_stability:.3f}")
            st.metric("Предсказанный FAR", f"{training_result.quality.predicted_far:.2e}")
            if training_result.quality.warnings:
                for w in training_result.quality.warnings:
                    st.warning(w)

            # Show stability chart
            from ..components.stability_chart import render_stability_chart
            render_stability_chart(training_result.stability_own)
    else:
        st.info(f"Нужно ещё {progress.samples_required - progress.samples_collected} образцов.")
