"""
Вход (аутентификация) — вкладка 4.

  1. Выбор профиля
  2. Сбор биометрии (камера / файл / датасет)
  3. НПБК.authenticate() → если принят → расшифровать пассфразу Кузнечиком
  4. Показать результат
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from ...config import AppConfig
from ...core.authentication import AuthenticationSession
from ...crypto import decrypt_passphrase
from ...datasets import CREMADAdapter, MEADAdapter, OULUAdapter, RAVDESSAdapter
from ...db import ProfileRepo
from ..components.pipeline_viz import render_stage_output
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


def render(repo: ProfileRepo, config: AppConfig) -> None:
    st.title("🔓 Вход")

    profiles = repo.list_profiles()
    if not profiles:
        st.info("Профилей нет. Сначала зарегистрируйся на вкладке **📝 Регистрация**.")
        return

    # Если уже есть результат — показываем его
    if st.session_state.get("_login_done"):
        _show_login_result()
        if st.button("🔄 Попробовать снова"):
            for k in list(st.session_state.keys()):
                if k.startswith("_login"):
                    del st.session_state[k]
            st.rerun()
        return

    # ================================================================ Выбор профиля
    user_id = st.selectbox("Профиль", profiles, key="login_user_id")

    try:
        profile_row = repo.load_profile_row(user_id)
        st.caption(
            f"Эмоция при регистрации: **{profile_row.emotion}** | "
            f"Образцов: {profile_row.n_own_samples} | "
            f"FAR (обучение): {profile_row.empirical_far:.3f}"
        )
    except Exception:
        pass

    st.markdown("---")

    # ================================================================ Сбор биометрии
    st.markdown("### Захват биометрии")
    st.caption("Используй ту же эмоцию, что при регистрации.")

    tab_cam, tab_file, tab_ds = st.tabs(["📷 Камера", "📁 Файл", "📊 Датасет"])

    def _run_auth(frames_or_path, from_frames: bool = False):
        try:
            container = repo.load_profile(user_id)
            auth_session = AuthenticationSession(container, config)
            with st.spinner("Аутентификация..."):
                if from_frames:
                    auth_result = auth_session.authenticate_frames(frames_or_path)
                else:
                    auth_result = auth_session.authenticate(frames_or_path)

            # Логируем попытку
            repo.log_auth(
                user_id=user_id,
                accepted=auth_result.accepted,
                hamming=auth_result.auth_result.hamming_distance,
                confidence=auth_result.auth_result.confidence,
            )

            # Расшифровываем пассфразу если принят
            secret = None
            if auth_result.accepted:
                profile_row = repo.load_profile_row(user_id)
                if profile_row.enc_secret and profile_row.kdf_salt:
                    try:
                        secret = decrypt_passphrase(
                            container.reference_code,
                            profile_row.enc_secret,
                            profile_row.kdf_salt,
                        )
                    except ValueError as e:
                        secret = f"[ошибка расшифровки: {e}]"

            st.session_state["_login_done"] = True
            st.session_state["_login_result"] = auth_result
            st.session_state["_login_secret"] = secret
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")

    with tab_cam:
        frames = webcam_recorder(key="login_cam")
        if frames:
            _run_auth(frames, from_frames=True)

    with tab_file:
        uploaded = st.file_uploader("Загрузи видео", type=["mp4", "avi", "mov"], key="login_file")
        if uploaded and st.button("Войти с видео", key="login_file_btn"):
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
                f.write(uploaded.read())
                tmp = f.name
            try:
                _run_auth(tmp)
            finally:
                os.unlink(tmp)

    with tab_ds:
        adapters = _get_adapters()
        if adapters:
            ds_name = st.selectbox("Датасет", list(adapters.keys()), key="login_ds")
            adapter = adapters[ds_name]
            try:
                profile_row = repo.load_profile_row(user_id)
                emotion = profile_row.emotion
            except Exception:
                emotion = adapter.emotions()[0]

            subjects = adapter.subjects()
            subj = st.selectbox("Актор (тест)", subjects, key="login_ds_subj")
            recs = adapter.recordings(subj, emotion.lower())
            if recs:
                rec_labels = [r.label for r in recs]
                sel = st.selectbox("Запись", rec_labels, key="login_ds_rec")
                rec = next(r for r in recs if r.label == sel)
                if st.button("▶ Войти с записью", key="login_ds_btn"):
                    _run_auth(rec.path)
            else:
                st.info(f"Нет записей {subj} / {emotion}")


def _show_login_result() -> None:
    auth_result = st.session_state.get("_login_result")
    secret = st.session_state.get("_login_secret")

    if auth_result is None:
        return

    npbk = auth_result.auth_result
    if auth_result.accepted:
        st.success("✅ ДОСТУП РАЗРЕШЁН")
        col1, col2, col3 = st.columns(3)
        col1.metric("Расстояние Хэмминга", npbk.hamming_distance)
        col2.metric("Порог", npbk.details.get("hamming_threshold", "—"))
        col3.metric("Уверенность", f"{npbk.confidence:.1%}")

        if secret:
            st.markdown("### 🔑 Расшифрованная пассфраза")
            st.code(secret, language=None)
            st.caption(
                "Пассфраза расшифрована ключом, производным от вашего бинарного кода НПБК "
                "(алгоритм Кузнечик, ГОСТ Р 34.12-2015)."
            )
        else:
            st.info("Пассфраза не была задана при регистрации.")
    else:
        st.error("❌ ДОСТУП ЗАПРЕЩЁН")
        col1, col2 = st.columns(2)
        col1.metric("Расстояние Хэмминга", npbk.hamming_distance)
        col2.metric("Порог", npbk.details.get("hamming_threshold", "—"))
        st.caption(
            "Расстояние Хэмминга превышает допустимый порог. "
            "Попробуй снова с той же эмоцией, что при регистрации."
        )
