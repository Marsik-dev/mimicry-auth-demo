"""
Регистрация пользователя — вкладка 3.

Шаги:
  1. Имя профиля + пассфраза + эмоция
  2. Сбор образцов (камера / файл / датасет)
  3. Обучение НПБК, шифрование пассфразы Кузнечиком, сохранение в БД
  4. Показ метрик FAR/FRR по слоям + код
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from ...config import AppConfig, DEFAULT_EMOTION_POOL
from ...core.enrollment import EnrollmentSession
from ...crypto import encrypt_passphrase
from ...datasets import CREMADAdapter, MEADAdapter, OULUAdapter, RAVDESSAdapter
from ...db import ProfileRepo
from ..components.layer_report import render_layer_reports
from ..components.pipeline_viz import render_stage_output
from ..components.realtime_processor import realtime_pipeline_widget
from ..components.stability_chart import render_stability_chart
from ..components.webcam_capture import webcam_recorder

DATASET_ROOT = Path(os.environ.get("DATASET_PATH", Path(__file__).parents[5] / "datasets"))


def _get_adapters():
    adapters = {}
    for name, cls in [("RAVDESS", RAVDESSAdapter), ("OULU-CASIA", OULUAdapter),
                      ("MEAD", MEADAdapter), ("CREMA-D", CREMADAdapter)]:
        a = cls(DATASET_ROOT)
        if a.available:
            adapters[name] = a
    return adapters


def _load_other_vectors(emotion: str) -> np.ndarray | None:
    """
    Загружаем «чужие» векторы из предобработанного датасета.
    Если есть ravdess_vectors.npz — берём ТОЛЬКО ту же эмоцию (ключ: *__emotion).
    Иначе — fallback на other_vectors.npz.
    """
    datasets_root = Path(os.environ.get("DATASET_PATH", Path(__file__).parents[6] / "datasets"))
    ravdess_npz = datasets_root / "processed" / "ravdess_vectors.npz"
    other_npz = Path(__file__).parents[4] / "assets" / "other_vectors.npz"

    if ravdess_npz.exists():
        try:
            data = np.load(ravdess_npz)
            emo_keys = [k for k in data.keys() if k.endswith(f"__{emotion}")]
            if emo_keys:
                return np.vstack([data[k] for k in emo_keys]).astype(np.float32)
        except Exception:
            pass

    if other_npz.exists():
        try:
            data = np.load(other_npz)
            return data["vectors"].astype(np.float32)
        except Exception:
            pass

    return None


def render(repo: ProfileRepo, config: AppConfig) -> None:
    st.title("📝 Регистрация пользователя")

    # Если регистрация уже завершена — показываем только результат
    if st.session_state.get("_reg_done"):
        _show_result()
        if st.button("🔄 Новая регистрация"):
            for k in list(st.session_state.keys()):
                if k.startswith("_reg") or k.startswith("reg_"):
                    del st.session_state[k]
            st.rerun()
        return

    # ================================================================ Шаг 1: Параметры
    st.markdown("### 1. Параметры профиля")
    col_id, col_pass, col_emo = st.columns([2, 2, 1])
    with col_id:
        user_id = st.text_input("Имя профиля", value="user1", key="reg_user_id")
    with col_pass:
        passphrase = st.text_input(
            "Пассфраза (секрет)", type="password", key="reg_passphrase",
            help="Эта строка будет зашифрована Кузнечиком и сохранена в БД. "
                 "При аутентификации — расшифрована и возвращена вам."
        )
    with col_emo:
        emotion = st.selectbox("Эмоция", config.emotion_pool, key="reg_emotion")

    if not user_id.strip():
        st.warning("Введи имя профиля.")
        return

    # Создаём/переиспользуем сессию обучения
    session_key = f"_reg_{user_id}_{emotion}"
    if st.session_state.get("_reg_session_key") != session_key:
        st.session_state["reg_session"] = EnrollmentSession(config, user_id)
        st.session_state["_reg_session_key"] = session_key
    session: EnrollmentSession = st.session_state["reg_session"]
    progress = session.get_progress()

    st.markdown("---")

    # ================================================================ Шаг 2: Образцы
    st.markdown(f"### 2. Образцы мимики — эмоция: **{emotion}**")
    st.progress(
        progress.progress_pct,
        text=f"Собрано: {progress.samples_collected} / {progress.samples_required}",
    )

    with st.expander("🔍 Pipeline Inspector (для отладки)", expanded=False):
        realtime_pipeline_widget(key="reg_live", initial_mode="landmark_extractor")

    tab_cam, tab_file, tab_ds = st.tabs(["📷 Камера", "📁 Файл", "📊 Датасет"])

    # Камера
    with tab_cam:
        frames = webcam_recorder(key="reg_cam")
        if frames:
            with st.spinner("Извлекаю признаки..."):
                result = session.process_frames(frames)
            if result.feature_vector is not None and result.quality_score > 0:
                session.add_sample(result)
                st.success(f"Добавлен образец (качество: {result.quality_score:.2f})")
                render_stage_output(result.stage_output)
                st.rerun()
            else:
                st.error("Не удалось извлечь признаки. Убедись что лицо в кадре.")

    # Файл
    with tab_file:
        uploaded = st.file_uploader("Загрузи видео", type=["mp4", "avi", "mov"], key="reg_file")
        if uploaded and st.button("Добавить образец из файла", key="reg_file_btn"):
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            try:
                with st.spinner("Извлекаю признаки..."):
                    result = session.process_video(tmp_path)
                if result.feature_vector is not None and result.quality_score > 0:
                    session.add_sample(result)
                    st.success(f"Добавлен образец (качество: {result.quality_score:.2f})")
                    st.rerun()
                else:
                    st.error("Не удалось извлечь признаки.")
            finally:
                os.unlink(tmp_path)

    # Датасет
    with tab_ds:
        adapters = _get_adapters()
        if adapters:
            ds_name = st.selectbox("Датасет", list(adapters.keys()), key="reg_ds")
            adapter = adapters[ds_name]
            subjects = adapter.subjects()
            subj = st.selectbox("Актор (будет «Свой»)", subjects, key="reg_ds_subj")
            recs = adapter.recordings(subj, emotion.lower())
            if recs:
                st.caption(f"Найдено {len(recs)} записей — добавим все.")
                if st.button(f"Добавить {len(recs)} образцов из датасета", key="reg_ds_btn"):
                    added = 0
                    errors = []
                    prog = st.progress(0)
                    for i, rec in enumerate(recs):
                        try:
                            result = session.process_video(rec.path)
                            if result.feature_vector is not None and result.quality_score > 0:
                                session.add_sample(result)
                                added += 1
                        except Exception as e:
                            errors.append(str(e))
                        prog.progress((i + 1) / len(recs))
                    prog.empty()
                    st.success(f"Добавлено {added} образцов")
                    if errors:
                        st.warning(f"Ошибки ({len(errors)}): {errors[0]}")
                    st.rerun()
            else:
                st.info(f"Нет записей для {subj} / {emotion}")

    # ================================================================ Шаг 3: Регистрация
    st.markdown("---")
    st.markdown("### 3. Обучить НПБК и зарегистрировать")

    if not progress.is_ready:
        st.info(
            f"Нужно ещё {progress.samples_required - progress.samples_collected} образцов "
            f"(или {progress.samples_required} — ГОСТ Р 52633.5). "
            "Недостающие будут сгенерированы аугментацией."
        )

    if not passphrase:
        st.warning("Введи пассфразу (шаг 1) перед регистрацией.")
    elif st.button("🔐 Зарегистрировать", type="primary", key="reg_submit",
                   disabled=not progress.is_ready and progress.samples_collected == 0):
        other_vecs = _load_other_vectors(emotion)
        if other_vecs is None or len(other_vecs) < 64:
            st.error(
                "Не удалось загрузить векторы «Чужих» (нужно ≥ 64). "
                "Убедись что датасет обработан (datasets/processed/ravdess_vectors.npz) "
                "или есть assets/other_vectors.npz."
            )
            return

        # Загружаем нормализатор для этой эмоции (если есть)
        normalizer_id = repo.latest_normalizer_id(emotion)
        normalizer_obj = None
        if normalizer_id is not None:
            try:
                normalizer_obj = repo.load_normalizer(normalizer_id)
            except Exception:
                normalizer_id = None

        with st.spinner("Обучаю НПБК..."):
            try:
                result, container = session.train(other_vecs)
            except Exception as e:
                st.error(f"Ошибка обучения: {e}")
                return

        # Шифруем пассфразу
        import os as _os
        salt = _os.urandom(16)
        enc_secret = encrypt_passphrase(container.reference_code, passphrase, salt)

        # Сохраняем в БД
        repo.save_profile(
            user_id=user_id.strip(),
            container=container,
            emotion=emotion,
            enc_secret=enc_secret,
            kdf_salt=salt,
            normalizer_id=normalizer_id,
        )

        # Сохраняем результат для отображения
        st.session_state["_reg_result"] = result
        st.session_state["_reg_container"] = container
        st.session_state["_reg_user_id"] = user_id
        st.session_state["_reg_done"] = True
        st.rerun()


def _show_result() -> None:
    result = st.session_state.get("_reg_result")
    container = st.session_state.get("_reg_container")
    user_id = st.session_state.get("_reg_user_id", "")

    st.success(f"✅ Пользователь **{user_id}** зарегистрирован!")
    st.caption("Пассфраза зашифрована Кузнечиком (ГОСТ Р 34.12-2015) и сохранена в БД.")

    if result is None or container is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Длина кода (бит)", len(container.reference_code))
    col2.metric("Порог Хэмминга", container.hamming_threshold)
    col3.metric("Ср. стабильность", f"{container.quality.mean_stability:.3f}")
    col4.metric("Эмпирический FAR", f"{container.quality.predicted_far:.3f}")

    render_layer_reports(result)

    with st.expander("Стабильность нейронов (слой 2)", expanded=False):
        render_stability_chart(result.stability_own, title="Стабильность нейронов (Own)")

    if result.quality.warnings:
        for w in result.quality.warnings:
            st.warning(w)
