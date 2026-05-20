"""Analysis page — stability chart and FAR/FRR for enrolled profiles."""
from __future__ import annotations

import numpy as np
import streamlit as st

from ...storage.profile_store import ProfileStore
from ..components.stability_chart import render_stability_chart
from npbk import NBKContainer
from npbk.metrics import evaluate


def render(store: ProfileStore) -> None:
    st.title("Анализ качества")

    profiles = store.list_profiles()
    if not profiles:
        st.info("Нет профилей для анализа.")
        return

    user_id = st.selectbox("Профиль", profiles)
    container = store.load_profile(user_id)

    st.subheader("Метрики обучения")
    q = container.quality
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средняя стабильность", f"{q.mean_stability:.3f}")
    col2.metric("Мин. стабильность", f"{q.min_stability:.3f}")
    col3.metric("Предсказанный FAR", f"{q.predicted_far:.2e}")
    col4.metric("Предсказанный FRR", f"{q.predicted_frr:.2%}")

    if q.warnings:
        for w in q.warnings:
            st.warning(w)

    st.markdown("---")
    st.subheader("Стабильность нейронов")
    render_stability_chart(
        np.ones(len(container.reference_code)) * 0.8,  # placeholder — need stored stability
        threshold=0.6,
        title=f"Stability ω_i — {user_id} ({len(container.reference_code)} нейронов)",
    )

    st.markdown("---")
    st.subheader("Информация о контейнере")
    st.json({
        "version": container.version,
        "created_at": container.created_at,
        "feature_dim": container.feature_dim,
        "code_length": len(container.reference_code),
        "hamming_threshold": container.hamming_threshold,
        "n_own_samples": q.n_own_samples,
        "n_other_samples": q.n_other_samples,
    })
