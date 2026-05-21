"""Streamlit application entry point — 4-page navigation."""
from __future__ import annotations

import streamlit as st

from .config import AppConfig
from .db import ProfileRepo, init_db
from .ui.pages import inspector, login, normalizer_training, registration


def main() -> None:
    st.set_page_config(
        page_title="Mimicry Auth",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Инициализируем БД (создаём таблицы если их нет)
    try:
        init_db()
    except Exception as e:
        st.sidebar.warning(f"⚠ БД недоступна: {e}")

    config = AppConfig.load()
    repo = ProfileRepo()

    with st.sidebar:
        st.markdown("## 🔐 Mimicry Auth")
        st.markdown("*ГОСТ Р 52633.5-2011*")
        st.markdown("---")
        page = st.radio(
            "Навигация",
            [
                "🔍 Pipeline Inspector",
                "⚙️ Нормализатор",
                "📝 Регистрация",
                "🔓 Вход",
            ],
            key="nav_page",
        )
        st.markdown("---")

        # Статистика профилей в sidebar
        try:
            infos = repo.list_profile_infos()
            st.caption(f"Профилей в БД: {len(infos)}")
        except Exception:
            st.caption("Профили недоступны")

    if page == "🔍 Pipeline Inspector":
        inspector.render(config)
    elif page == "⚙️ Нормализатор":
        normalizer_training.render(repo, config)
    elif page == "📝 Регистрация":
        registration.render(repo, config)
    elif page == "🔓 Вход":
        login.render(repo, config)


if __name__ == "__main__":
    main()
