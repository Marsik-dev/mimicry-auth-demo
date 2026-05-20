"""
Reusable source selector component.

Shows tabs: [🔴 Live] [📷 Record] [📁 File] [📊 Dataset]
The Dataset tab allows picking a subject + emotion from RAVDESS/OULU-CASIA
and returns their video paths as a list, just like the other sources.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from ..pages.dataset_browser import DATASET_ROOT, OULUAdapter, RAVDESSAdapter, Recording
from .realtime_processor import realtime_pipeline_widget
from .webcam_capture import webcam_recorder


def _get_adapters():
    adapters = {}
    r, o = RAVDESSAdapter(DATASET_ROOT), OULUAdapter(DATASET_ROOT)
    if r.available:
        adapters["RAVDESS"] = r
    if o.available:
        adapters["OULU-CASIA"] = o
    return adapters


def source_selector(
    key: str,
    pipeline,
    live_mode: str = "landmark_extractor",
    show_live: bool = True,
) -> tuple[list[np.ndarray] | None, list[Path] | None]:
    """
    Renders source tabs and returns one of:
      (frames: list[ndarray], None)  — webcam/file, frames ready for pipeline
      (None, paths: list[Path])      — dataset paths, caller runs pipeline on each

    Returns (None, None) when nothing is ready yet.
    """

    tabs = []
    tab_keys = []
    if show_live:
        tabs.append("🔴 Live")
        tab_keys.append("live")
    tabs += ["📷 Запись", "📁 Файл", "📊 Датасет"]
    tab_keys += ["cam", "file", "dataset"]

    tab_objs = st.tabs(tabs)
    tab_map = dict(zip(tab_keys, tab_objs))

    # --- Live ---
    if "live" in tab_map:
        with tab_map["live"]:
            realtime_pipeline_widget(key=f"{key}_live", initial_mode=live_mode)

    # --- Record ---
    with tab_map["cam"]:
        frames = webcam_recorder(key=f"{key}_cam")
        if frames is not None:
            return frames, None

    # --- File ---
    with tab_map["file"]:
        uploaded = st.file_uploader(
            "Видео файл", type=["mp4", "avi", "mov"], key=f"{key}_upload"
        )
        if uploaded and st.button("Обработать", key=f"{key}_file_btn"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(uploaded.read())
                tmp = f.name
            frames_from_file = pipeline.run_from_frames_path(tmp)
            os.unlink(tmp)
            # wrap raw video as frames via pipeline
            return frames_from_file, None

    # --- Dataset ---
    with tab_map["dataset"]:
        adapters = _get_adapters()
        if not adapters:
            st.warning("Датасеты не найдены. Скачай RAVDESS или OULU-CASIA.")
            return None, None

        ds_name = st.selectbox("Датасет", list(adapters.keys()), key=f"{key}_ds")
        adapter = adapters[ds_name]
        subjects = adapter.subjects()
        emotions = adapter.emotions()

        col_s, col_e = st.columns(2)
        with col_s:
            subject = st.selectbox("Субъект", subjects, key=f"{key}_subj")
        with col_e:
            emotion = st.selectbox("Эмоция", emotions, key=f"{key}_emo")

        recs: list[Recording] = adapter.recordings(subject, emotion)
        if not recs:
            st.info(f"Нет записей: {subject} / {emotion}")
            return None, None

        st.success(f"Найдено {len(recs)} записей — {subject} / {emotion}")

        # Show thumbnails
        if st.checkbox("Предпросмотр записей", key=f"{key}_preview"):
            import cv2
            cols = st.columns(min(4, len(recs)))
            for col, rec in zip(cols, recs[:4]):
                cap = cv2.VideoCapture(str(rec.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    import cv2 as cv
                    col.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                              caption=rec.label[:20], use_container_width=True)

        if st.button(f"Использовать все {len(recs)} записей", key=f"{key}_use",
                     type="primary"):
            return None, [rec.path for rec in recs]

    return None, None
