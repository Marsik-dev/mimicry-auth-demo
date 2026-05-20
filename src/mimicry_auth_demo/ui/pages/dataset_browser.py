"""
Dataset Browser — browse RAVDESS or OULU-CASIA recordings.

Allows selecting a subject (random or by ID), picking an emotion,
then viewing individual recordings and optionally using them as
enrollment samples.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import streamlit as st

from ...config import AppConfig
from ...storage.profile_store import ProfileStore

DATASET_ROOT = Path(os.environ.get("DATASET_PATH", Path(__file__).parent.parent.parent.parent.parent.parent / "datasets"))


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    path: Path
    subject_id: str
    emotion: str
    label: str   # display name for UI (e.g. "Actor_01 / happy / clip_001")


class RAVDESSAdapter:
    EMOTIONS = {
        "1": "neutral", "2": "calm", "3": "happy", "4": "sad",
        "5": "angry", "6": "fear", "7": "disgust", "8": "surprise",
    }

    def __init__(self, root: Path) -> None:
        self.root = root / "ravdess" / "videos"

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def subjects(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def emotions(self) -> list[str]:
        return sorted(set(self.EMOTIONS.values()))

    def recordings(self, subject: str, emotion: str) -> list[Recording]:
        subj_dir = self.root / subject
        if not subj_dir.is_dir():
            return []
        # Find emotion number
        emo_num = next((k for k, v in self.EMOTIONS.items() if v == emotion), None)
        if emo_num is None:
            return []
        result = []
        for mp4 in sorted(subj_dir.glob("*.mp4")):
            parts = mp4.stem.split("-")
            if len(parts) == 7 and parts[0] == "02" and parts[2] == emo_num:
                result.append(Recording(
                    path=mp4,
                    subject_id=subject,
                    emotion=emotion,
                    label=mp4.name,
                ))
        return result


class OULUAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root / "oulu"

    @property
    def available(self) -> bool:
        return self.root.is_dir() and any(self.root.glob("Subject_*"))

    def subjects(self) -> list[str]:
        return sorted(d.name for d in self.root.glob("Subject_*") if d.is_dir())

    def emotions(self) -> list[str]:
        emotions: set[str] = set()
        for subj in self.root.glob("Subject_*"):
            for emo in subj.iterdir():
                if emo.is_dir():
                    emotions.add(emo.name)
        return sorted(emotions)

    def recordings(self, subject: str, emotion: str) -> list[Recording]:
        emo_dir = self.root / subject / emotion
        if not emo_dir.is_dir():
            return []
        result = []
        for vid in sorted(emo_dir.glob("*.avi")) + sorted(emo_dir.glob("*.mp4")):
            result.append(Recording(
                path=vid,
                subject_id=subject,
                emotion=emotion,
                label=vid.name,
            ))
        return result


def _get_adapters() -> dict[str, RAVDESSAdapter | OULUAdapter]:
    adapters: dict[str, RAVDESSAdapter | OULUAdapter] = {}
    r = RAVDESSAdapter(DATASET_ROOT)
    o = OULUAdapter(DATASET_ROOT)
    if r.available:
        adapters["RAVDESS"] = r
    if o.available:
        adapters["OULU-CASIA"] = o
    return adapters


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def _frame_grid(path: Path, n_frames: int = 6) -> None:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // n_frames)
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        cols = st.columns(len(frames))
        for col, frame in zip(cols, frames):
            col.image(frame, use_container_width=True)


def render(store: ProfileStore, config: AppConfig) -> None:
    st.title("Dataset Browser")
    st.markdown("Просматривай записи из датасета и используй их как образцы для регистрации.")

    adapters = _get_adapters()
    if not adapters:
        st.error(
            f"Датасеты не найдены в `{DATASET_ROOT}`. "
            "Скачай RAVDESS или OULU-CASIA и положи в папку `datasets/`."
        )
        return

    # --- Dataset selector ---
    dataset_name = st.selectbox("Датасет", list(adapters.keys()))
    adapter = adapters[dataset_name]

    subjects = adapter.subjects()
    emotions = adapter.emotions()

    if not subjects:
        st.warning("Нет субъектов в датасете.")
        return

    # --- Subject selector ---
    col_subj, col_emo = st.columns(2)

    with col_subj:
        st.markdown("**Субъект**")
        if st.button("🎲 Случайный субъект"):
            st.session_state["browser_subject"] = random.choice(subjects)

        subject = st.selectbox(
            "Выбрать субъекта",
            subjects,
            index=subjects.index(st.session_state.get("browser_subject", subjects[0]))
            if st.session_state.get("browser_subject") in subjects else 0,
            key="browser_subject_sel",
            label_visibility="collapsed",
        )
        st.session_state["browser_subject"] = subject

    with col_emo:
        st.markdown("**Эмоция**")
        emotion = st.selectbox("Выбрать эмоцию", emotions, key="browser_emotion_sel",
                               label_visibility="collapsed")

    # --- Recordings ---
    recordings = adapter.recordings(subject, emotion)

    if not recordings:
        st.info(f"Нет записей: {subject} / {emotion}")
        return

    st.markdown(f"**Найдено {len(recordings)} записей** — {subject} / {emotion}")

    col_rec, col_rand = st.columns([3, 1])
    with col_rec:
        labels = [r.label for r in recordings]
        sel_label = st.selectbox("Запись", labels, key="browser_rec_sel")
    with col_rand:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🎲 Случайная"):
            sel_label = random.choice(labels)
            st.session_state["browser_rec_sel"] = sel_label
            st.rerun()

    rec = next(r for r in recordings if r.label == sel_label)

    # --- Preview ---
    st.markdown("---")
    st.subheader(f"Предпросмотр: `{rec.label}`")

    tab_frames, tab_video = st.tabs(["🖼 Кадры", "▶ Видео"])

    with tab_frames:
        _frame_grid(rec.path, n_frames=6)

    with tab_video:
        st.video(str(rec.path))

    # --- Use as enrollment sample ---
    st.markdown("---")
    if store.list_profiles():
        user_id = st.selectbox("Добавить в сессию регистрации для:", store.list_profiles(),
                               key="browser_enroll_uid")
    else:
        user_id = st.text_input("Имя пользователя для регистрации", value="user1")

    if st.button("📥 Использовать как образец для регистрации", type="primary"):
        from ...core.enrollment import EnrollmentSession

        sess_key = f"enroll_session_{user_id}"
        if sess_key not in st.session_state:
            st.session_state[sess_key] = EnrollmentSession(config, user_id)
        session: EnrollmentSession = st.session_state[sess_key]

        with st.spinner("Обработка видео..."):
            result = session.process_video(str(rec.path))
            session.add_sample(result)

        progress = session.get_progress()
        st.success(
            f"✅ Образец добавлен! "
            f"Собрано: {progress.samples_collected}/{progress.samples_required}. "
            f"Качество: {result.quality_score:.3f}"
        )
        st.info("Перейди на страницу **Регистрация** чтобы завершить обучение НПБК.")
