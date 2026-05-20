"""Streamlit application entry point."""
from __future__ import annotations

import streamlit as st

from .config import AppConfig
from .storage.profile_store import ProfileStore
from .ui.pages import analysis, authentication, enrollment, home, pipeline_inspector


def main() -> None:
    st.set_page_config(
        page_title="Mimicry Auth Demo",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = AppConfig.load()
    store = ProfileStore(config.store_dir)

    with st.sidebar:
        st.markdown("## 🔐 Mimicry Auth")
        st.markdown("ГОСТ Р 52633.5-2011")
        st.markdown("---")
        page = st.radio(
            "Навигация",
            ["Главная", "Регистрация", "Аутентификация", "Pipeline Inspector", "Анализ"],
            key="nav_page",
        )

    if page == "Главная":
        home.render(store)
    elif page == "Регистрация":
        enrollment.render(store, config)
    elif page == "Аутентификация":
        authentication.render(store, config)
    elif page == "Pipeline Inspector":
        pipeline_inspector.render(config)
    elif page == "Анализ":
        analysis.render(store)


if __name__ == "__main__":
    main()
