"""
Pipeline Inspector — вкладка 1.

Источники:
  - Камера в реальном времени (live overlay)
  - Запись с веб-камеры
  - Загрузка файла
  - Видео из датасета (RAVDESS / OULU / MEAD / CREMA-D)

Показывает результат каждого этапа предобработки.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from ...config import AppConfig
from ...datasets import CREMADAdapter, MEADAdapter, OULUAdapter, RAVDESSAdapter
from ..components.pipeline_settings import export_config_button, get_current_config, render_stage_settings
from ..components.pipeline_viz import STAGE_NAMES
from ..components.pipeline_viz import render_stage_output
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.webcam_capture import webcam_recorder

DATASET_ROOT = Path(os.environ.get("DATASET_PATH", Path(__file__).parents[5] / "datasets"))


def _get_adapters():
    adapters = {}
    for name, cls, subdir in [
        ("RAVDESS", RAVDESSAdapter, None),
        ("OULU-CASIA", OULUAdapter, None),
        ("MEAD", MEADAdapter, None),
        ("CREMA-D", CREMADAdapter, None),
    ]:
        adapter = cls(DATASET_ROOT)
        if adapter.available:
            adapters[name] = adapter
    return adapters


def render(config: AppConfig) -> None:
    st.title("🔍 Pipeline Inspector")
    st.caption(
        "Выбери источник видео и посмотри что происходит на каждом этапе предобработки."
    )

    # Настройки pipeline в expander с выбором этапа
    with st.expander("⚙️ Параметры pipeline", expanded=False):
        col_stage, col_settings = st.columns([1, 3])
        with col_stage:
            selected_stage = st.radio("Этап", STAGE_NAMES, index=4, key="insp_stage")
        with col_settings:
            render_stage_settings(selected_stage)
        export_config_button(get_current_config())
    pipeline_cfg = get_current_config()

    st.markdown("---")

    tab_live, tab_cam, tab_file, tab_ds = st.tabs([
        "🔴 Реальное время", "📷 Запись", "📁 Файл", "📊 Датасет"
    ])

    # ---------------------------------------------------------------- Live
    with tab_live:
        st.caption("Наведи камеру на лицо. Все этапы отображаются поверх изображения.")
        realtime_pipeline_widget(key="insp_live", initial_mode="landmark_extractor")

    # ---------------------------------------------------------------- Webcam record
    with tab_cam:
        st.caption("Нажми **Начать** и запиши 3–5 секунд. Затем **Остановить**.")
        frames = webcam_recorder(key="insp_cam")
        if frames:
            from mimicry_preproc import Pipeline

            pipeline = Pipeline(pipeline_cfg)
            with st.spinner("Обрабатываю..."):
                stage_out = pipeline.run_with_debug_from_frames(frames)
            st.success(f"Обработано {len(frames)} кадров")
            for s in STAGE_NAMES:
                render_stage_output(s, stage_out)

    # ---------------------------------------------------------------- File upload
    with tab_file:
        uploaded = st.file_uploader("Загрузи видео", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            try:
                from mimicry_preproc import Pipeline

                pipeline = Pipeline(pipeline_cfg)
                with st.spinner("Обрабатываю..."):
                    stage_out = pipeline.run_with_debug(tmp_path)
                for s in STAGE_NAMES:
                    render_stage_output(s, stage_out)
            finally:
                os.unlink(tmp_path)

    # ---------------------------------------------------------------- Dataset
    with tab_ds:
        adapters = _get_adapters()
        if not adapters:
            st.warning(
                f"Датасеты не найдены в `{DATASET_ROOT}`. "
                "Убедись что датасет смонтирован (Docker: DATASET_PATH=/data/datasets)."
            )
            return

        ds_name = st.selectbox("Датасет", list(adapters.keys()), key="insp_ds")
        adapter = adapters[ds_name]

        col_subj, col_emo = st.columns(2)
        with col_subj:
            subjects = adapter.subjects()
            subject = st.selectbox("Актор", subjects, key="insp_subj")
        with col_emo:
            emotions = adapter.emotions()
            emotion = st.selectbox("Эмоция", emotions, key="insp_emo")

        recordings = adapter.recordings(subject, emotion.lower())
        if not recordings:
            st.info("Нет записей для выбранной комбинации.")
            return

        rec_labels = [r.label for r in recordings]
        sel_label = st.selectbox("Запись", rec_labels, key="insp_rec")
        rec = next(r for r in recordings if r.label == sel_label)

        # Frame preview
        cap = cv2.VideoCapture(str(rec.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // 5)
        frames_preview = []
        for i in range(5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frames_preview.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames_preview:
            cols = st.columns(len(frames_preview))
            for c, f in zip(cols, frames_preview):
                c.image(f, use_container_width=True)

        if st.button("▶ Запустить Pipeline", key="insp_run"):
            from mimicry_preproc import Pipeline

            pipeline = Pipeline(pipeline_cfg)
            with st.spinner("Обрабатываю..."):
                stage_out = pipeline.run_with_debug(rec.path)
            for s in STAGE_NAMES:
                render_stage_output(s, stage_out)
