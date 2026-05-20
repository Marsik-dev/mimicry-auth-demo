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
from ..pages.dataset_browser import DATASET_ROOT, OULUAdapter, RAVDESSAdapter


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Аутентификация")

    profiles = store.list_profiles()
    if not profiles:
        st.warning("Нет зарегистрированных профилей. Перейдите в **Регистрация**.")
        return

    user_id = st.selectbox("Профиль для проверки", profiles)
    container = store.load_profile(user_id)

    tab_live, tab_cam, tab_file, tab_ds = st.tabs([
        "🔴 Live preview", "📷 Запись", "📁 Файл", "📊 Из датасета"
    ])

    result = None

    with tab_live:
        st.caption("Убедись что лицо детектируется перед записью.")
        realtime_pipeline_widget(key="auth_live", initial_mode="face_detector")

    with tab_cam:
        frames = webcam_recorder(key="auth_webcam")
        if frames is not None:
            from ...core.authentication import AuthenticationSession
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                result = AuthenticationSession(container, config).authenticate_frames(frames)

    with tab_file:
        uploaded = st.file_uploader("Видео", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Аутентифицировать"):
            from ...core.authentication import AuthenticationSession
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Обработка..."):
                result = AuthenticationSession(container, config).authenticate(tmp_path)
            os.unlink(tmp_path)

    with tab_ds:
        result = _dataset_auth_tab(container, config)

    if result is not None:
        _show_result(result, container)


def _dataset_auth_tab(container, config: AppConfig):
    """Pick a random recording from dataset and try to authenticate it."""
    from ...core.authentication import AuthenticationSession

    adapters = {}
    r, o = RAVDESSAdapter(DATASET_ROOT), OULUAdapter(DATASET_ROOT)
    if r.available:
        adapters["RAVDESS"] = r
    if o.available:
        adapters["OULU-CASIA"] = o

    if not adapters:
        st.warning("Датасеты не найдены.")
        return None

    ds_name = st.selectbox("Датасет", list(adapters.keys()), key="auth_ds_name")
    adapter = adapters[ds_name]

    col_s, col_e = st.columns(2)
    with col_s:
        subject = st.selectbox("Субъект", adapter.subjects(), key="auth_ds_subj")
    with col_e:
        emotion = st.selectbox("Эмоция", adapter.emotions(), key="auth_ds_emo")

    recs = adapter.recordings(subject, emotion)
    if not recs:
        st.info(f"Нет записей: {subject}/{emotion}")
        return None

    labels = [r.label for r in recs]
    selected = st.selectbox("Запись", labels, key="auth_ds_rec")
    rec = next(r for r in recs if r.label == selected)

    col_a, col_b = st.columns(2)
    if col_a.button("🔍 Аутентифицировать эту запись", type="primary"):
        with st.spinner("Обработка..."):
            return AuthenticationSession(container, config).authenticate(str(rec.path))

    if col_b.button("🎲 Случайная запись"):
        import random
        rec = random.choice(recs)
        with st.spinner("Обработка..."):
            return AuthenticationSession(container, config).authenticate(str(rec.path))

    return None


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
