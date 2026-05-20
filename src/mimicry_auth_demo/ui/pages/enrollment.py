"""Enrollment page.

Flow:
  1. Enter profile name + choose emotion
  2. Collect Own samples via Live/Webcam/File/Dataset
  3. If < 11 samples → augment to 11 (GOST R 52633.2)
  4. Train NPBK → show Layer 1 then Layer 2 FAR/FRR
  5. Save profile
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from mimicry_preproc import Pipeline

from ...config import AppConfig, DEFAULT_EMOTION_POOL
from ...core.enrollment import EnrollmentSession
from ...storage.profile_store import ProfileStore
from ..components.layer_report import render_layer_reports
from ..components.pipeline_viz import STAGE_NAMES, render_stage_output
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.stability_chart import render_stability_chart
from ..components.webcam_capture import webcam_recorder
from ..pages.dataset_browser import DATASET_ROOT, OULUAdapter, RAVDESSAdapter

_OTHER_VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../assets/other_vectors.npz"
)


def _load_other_vectors() -> np.ndarray | None:
    try:
        data = np.load(_OTHER_VECTORS_PATH)
        return data["vectors"]
    except Exception:
        return None


def _get_adapters():
    adapters = {}
    r, o = RAVDESSAdapter(DATASET_ROOT), OULUAdapter(DATASET_ROOT)
    if r.available:
        adapters["RAVDESS"] = r
    if o.available:
        adapters["OULU-CASIA"] = o
    return adapters


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Регистрация пользователя")

    # --- Profile name + emotion ---
    col_uid, col_emo = st.columns([2, 1])
    with col_uid:
        user_id = st.text_input("Имя профиля", value="user1", key="enroll_user_id")
    with col_emo:
        emotion = st.selectbox(
            "Эмоция",
            config.emotion_pool,
            key="enroll_emotion",
            help="Используй одну эмоцию для всех образцов",
        )

    session_key = f"_enroll_{user_id}_{emotion}"
    if st.session_state.get("_enroll_key") != session_key:
        st.session_state["enroll_session"] = EnrollmentSession(config, user_id)
        st.session_state["_enroll_key"] = session_key
        st.session_state["enroll_last_stage_out"] = None

    session: EnrollmentSession = st.session_state["enroll_session"]
    progress = session.get_progress()

    st.progress(
        progress.progress_pct,
        text=f"Образцов: {progress.samples_collected} / {progress.samples_required}",
    )
    if progress.sample_quality_scores:
        cols = st.columns(min(11, len(progress.sample_quality_scores)))
        for col, score in zip(cols, progress.sample_quality_scores):
            col.metric("Q", f"{score:.2f}")

    st.markdown("---")

    # ===================================================================
    # Source tabs
    # ===================================================================
    tab_live, tab_cam, tab_file, tab_ds = st.tabs([
        "🔴 Live preview", "📷 Запись", "📁 Файл", "📊 Из датасета"
    ])

    with tab_live:
        st.caption("Убедись что лицо хорошо детектируется перед записью.")
        realtime_pipeline_widget(key="enroll_live", initial_mode="landmark_extractor")

    with tab_cam:
        frames = webcam_recorder(key="enroll_webcam")
        if frames is not None:
            with st.spinner(f"Обработка {len(frames)} кадров..."):
                result = session.process_frames(frames)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            st.success(f"✅ +1 образец (качество {result.quality_score:.3f})")
            st.rerun()

    with tab_file:
        uploaded = st.file_uploader("Видео", type=["mp4", "avi", "mov"])
        if uploaded and st.button("Обработать файл"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Обработка..."):
                result = session.process_video(tmp_path)
                session.add_sample(result)
                st.session_state["enroll_last_stage_out"] = result.stage_output
            os.unlink(tmp_path)
            st.success(f"✅ +1 образец (качество {result.quality_score:.3f})")
            st.rerun()

    with tab_ds:
        _render_dataset_tab(session, config, emotion)

    # Pipeline inspector (collapsed)
    last_out = st.session_state.get("enroll_last_stage_out")
    if last_out is not None:
        with st.expander("🔍 Pipeline Inspector", expanded=False):
            stage = st.selectbox("Стадия", STAGE_NAMES, key="enroll_stage")
            render_stage_output(stage, last_out)

    # ===================================================================
    # Train
    # ===================================================================
    st.markdown("---")
    _render_train_section(session, store, user_id, config)


def _render_dataset_tab(session: EnrollmentSession, config: AppConfig, emotion: str) -> None:
    """Tab: pick actor from dataset, process all their recordings."""
    adapters = _get_adapters()
    if not adapters:
        st.warning("Датасеты не найдены в `datasets/`.")
        return

    ds_name = st.selectbox("Датасет", list(adapters.keys()), key="enroll_ds_name")
    adapter = adapters[ds_name]
    subjects = adapter.subjects()
    emotions_ds = adapter.emotions()

    col_s, col_e = st.columns(2)
    with col_s:
        subject = st.selectbox("Актор / субъект", subjects, key="enroll_ds_subj")
    with col_e:
        # Try to match the chosen emotion to dataset's naming
        default_emo = _match_emotion(emotion, emotions_ds)
        emo_idx = emotions_ds.index(default_emo) if default_emo in emotions_ds else 0
        ds_emotion = st.selectbox("Эмоция", emotions_ds, index=emo_idx, key="enroll_ds_emo")

    recs = adapter.recordings(subject, ds_emotion)
    if not recs:
        st.info(f"Нет записей: {subject} / {ds_emotion}")
        return

    n_real = len(recs)
    n_needed = config.n_required_own_samples
    n_aug = max(0, n_needed - n_real)

    # Preview
    import cv2
    preview_cols = st.columns(min(4, n_real))
    for col, rec in zip(preview_cols, recs[:4]):
        cap = cv2.VideoCapture(str(rec.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            col.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                      caption=rec.label[:18], use_container_width=True)

    st.markdown(f"**{n_real} реальных записей**")
    if n_aug > 0:
        st.info(
            f"После обработки будет ~{n_real * 2} образцов. "
            f"Если < {n_needed} — генерируем {n_aug} синтетических (ГОСТ Р 52633.2)."
        )
    else:
        st.success(f"Достаточно записей для ГОСТ (≥{n_needed}).")

    if st.button(f"📥 Обработать все {n_real} записи {subject}/{ds_emotion}",
                 type="primary", key="enroll_ds_process"):
        pipeline = Pipeline(config.pipeline)
        progress_bar = st.progress(0.0, text="Обработка видео...")
        processed = 0
        for i, rec in enumerate(recs):
            try:
                result = session.process_video(str(rec.path))
                session.add_sample(result)
                processed += 1
                st.session_state["enroll_last_stage_out"] = result.stage_output
            except Exception as e:
                st.warning(f"  ⚠ {rec.label}: {e}")
            progress_bar.progress((i + 1) / len(recs), text=f"{i+1}/{len(recs)} видео...")

        st.success(f"✅ {processed} образцов добавлено из {subject}/{ds_emotion}")
        st.rerun()


def _match_emotion(ui_emotion: str, dataset_emotions: list[str]) -> str:
    """Try to find the matching dataset emotion name for a UI emotion key."""
    mapping = {
        "Happy": ["happy", "Happiness", "happiness"],
        "Angry": ["angry", "Anger", "anger"],
        "Surprise": ["surprise", "Surprise", "surprised"],
        "Sad": ["sad", "Sadness", "sadness"],
        "Fear": ["fear", "Fear", "fearful"],
        "Disgust": ["disgust", "Disgust"],
    }
    candidates = mapping.get(ui_emotion, [ui_emotion.lower()])
    for c in candidates:
        if c in dataset_emotions:
            return c
    return dataset_emotions[0] if dataset_emotions else ui_emotion


def _render_train_section(
    session: EnrollmentSession,
    store: ProfileStore,
    user_id: str,
    config: AppConfig,
) -> None:
    progress = session.get_progress()
    n = progress.samples_collected
    req = progress.samples_required

    if n == 0:
        st.info("Добавь хотя бы один образец чтобы начать регистрацию.")
        return

    # Show augmentation notice if we'll need to generate samples
    if n < req:
        from npbk import augmentation_info
        st.warning(
            f"Собрано {n} образцов — меньше ГОСТ-минимума ({req}). "
            f"Будет применена аугментация: {augmentation_info(n, req)}"
        )
        btn_label = f"🧠 Обучить (с аугментацией до {req})"
    else:
        st.success(f"✅ Достаточно образцов ({n} ≥ {req}). Готово к обучению.")
        btn_label = "🧠 Обучить НПБК"

    if st.button(btn_label, type="primary", disabled=n == 0):
        other_vecs = _load_other_vectors()
        if other_vecs is None:
            st.error("Нет `assets/other_vectors.npz`. Запусти сборку датасета.")
            return

        with st.spinner("Обучение НПБК..."):
            training_result, container = session.train(other_vecs)

        store.save_profile(user_id, container)
        st.success(f"✅ Профиль «{user_id}» сохранён!")

        # Metrics summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Стабильность (итог)", f"{training_result.quality.mean_stability:.3f}")
        col2.metric("FAR (итог)", f"{training_result.quality.predicted_far:.2e}")
        col3.metric("Длина кода", len(training_result.reference_code))

        for w in training_result.quality.warnings:
            st.warning(w)

        # Per-layer reports
        render_layer_reports(training_result)

        # Overall stability chart
        with st.expander("График стабильности нейронов (итог)"):
            render_stability_chart(training_result.stability_own)
