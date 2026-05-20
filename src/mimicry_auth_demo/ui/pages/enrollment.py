"""Enrollment page: record facial expression videos, train NPBK, save profile."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st

from ...config import AppConfig, DEFAULT_EMOTION_POOL
from ...core.enrollment import EnrollmentSession
from ...storage.profile_store import ProfileStore
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.stability_chart import render_stability_chart
from ..components.webcam_capture import webcam_recorder

_OTHER_VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../assets/other_vectors.npz"
)


def _load_other_vectors() -> np.ndarray | None:
    try:
        data = np.load(_OTHER_VECTORS_PATH)
        return data["vectors"]
    except Exception:
        return None


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Регистрация пользователя")

    col_uid, col_emo = st.columns([2, 1])
    with col_uid:
        user_id = st.text_input("Имя пользователя", value="user1", key="enroll_user_id")
    with col_emo:
        emotion = st.selectbox(
            "Эмоция для регистрации",
            config.emotion_pool,
            key="enroll_emotion",
            help="Используй одну и ту же эмоцию для всех образцов. "
                 "Разные пользователи могут использовать разные эмоции из пула.",
        )

    session_key = f"enroll_session_{user_id}_{emotion}"
    if st.session_state.get("_enroll_key") != session_key:
        st.session_state["enroll_session"] = EnrollmentSession(config, user_id)
        st.session_state["_enroll_key"] = session_key
        st.session_state["enroll_last_stage_out"] = None

    session: EnrollmentSession = st.session_state["enroll_session"]
    progress = session.get_progress()

    st.progress(
        progress.progress_pct,
        text=f"Образцов собрано: {progress.samples_collected} / {progress.samples_required}",
    )

    if progress.sample_quality_scores:
        cols = st.columns(len(progress.sample_quality_scores))
        for col, score in zip(cols, progress.sample_quality_scores):
            col.metric("Качество", f"{score:.3f}")

    st.markdown("---")
    st.subheader("Запись образца")
    st.markdown(
        "Покажи одну эмоцию (улыбка, удивление, злость…) в течение **4–5 секунд**. "
        "Нужно собрать минимум **11 образцов**."
    )

    tab_live, tab_cam, tab_file = st.tabs(["🔴 Live preview", "📷 Запись", "📁 Файл"])

    with tab_live:
        st.markdown(
            "Живой предпросмотр — помогает убедиться что лицо детектируется "
            "до начала записи образца."
        )
        realtime_pipeline_widget(key="enroll_live", initial_mode="landmark_extractor")

    with tab_cam:
        frames = webcam_recorder(key="enroll_webcam", label="камера для регистрации")
        if frames is not None:
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                result = session.process_frames(frames)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            st.success(f"✅ Образец добавлен. Качество: {result.quality_score:.3f}")
            st.rerun()

    with tab_file:
        uploaded = st.file_uploader("Видео файл", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Обработать файл"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Обработка..."):
                result = session.process_video(tmp_path)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            os.unlink(tmp_path)
            st.success(f"✅ Образец добавлен. Качество: {result.quality_score:.3f}")
            st.rerun()

    # --- Pipeline inspector ---
    last_out = st.session_state.get("enroll_last_stage_out")
    if last_out is not None:
        st.markdown("---")
        with st.expander("Pipeline Inspector", expanded=False):
            selected_stage = st.selectbox(
                "Стадия", STAGE_NAMES, key="enroll_stage_sel"
            )
            render_stage_output(selected_stage, last_out)

    # --- Train ---
    st.markdown("---")
    if progress.is_ready:
        st.success(
            f"✅ Собрано {progress.samples_collected} образцов — "
            "можно обучить НПБК!"
        )
        if st.button("🧠 Обучить и сохранить профиль", type="primary"):
            other_vecs = _load_other_vectors()
            if other_vecs is None:
                st.error(
                    "Нет файла `assets/other_vectors.npz`. "
                    "Запусти скрипт обработки датасета: "
                    "`python datasets/build_dataset.py --segments 2`"
                )
                return
            with st.spinner("Обучение НПБК..."):
                training_result, container = session.train(other_vecs)
            store.save_profile(user_id, container)
            st.success(f"✅ Профиль «{user_id}» сохранён!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Стабильность", f"{training_result.quality.mean_stability:.3f}")
            col2.metric("FAR", f"{training_result.quality.predicted_far:.2e}")
            col3.metric("Нейронов", len(training_result.reference_code))
            for w in training_result.quality.warnings:
                st.warning(w)
            render_stability_chart(training_result.stability_own)
    else:
        st.info(
            f"Нужно ещё {progress.samples_required - progress.samples_collected} образцов."
        )
