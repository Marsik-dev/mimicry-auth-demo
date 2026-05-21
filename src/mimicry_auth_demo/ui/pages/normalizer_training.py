"""
Нормализатор — вкладка 2.

Обучает MinMax + PCA на случайных акторах из датасета для выбранной эмоции.
Нормализатор сохраняется в БД и применяется при регистрации / аутентификации.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ...config import AppConfig
from ...datasets import OULUAdapter, RAVDESSAdapter, MEADAdapter, CREMADAdapter
from ...db import ProfileRepo

DATASET_ROOT = Path(os.environ.get("DATASET_PATH", Path(__file__).parents[5] / "datasets"))


def _get_adapters():
    adapters = {}
    for name, cls in [("RAVDESS", RAVDESSAdapter), ("OULU-CASIA", OULUAdapter),
                      ("MEAD", MEADAdapter), ("CREMA-D", CREMADAdapter)]:
        a = cls(DATASET_ROOT)
        if a.available:
            adapters[name] = a
    return adapters


def _collect_vectors(adapter, subjects: list[str], emotion: str) -> np.ndarray | None:
    """Прогоняем записи через pipeline, возвращаем матрицу векторов."""
    from mimicry_preproc import Pipeline

    pipeline = Pipeline()
    vectors = []
    progress = st.progress(0, text="Собираю векторы...")
    for i, subj in enumerate(subjects):
        recs = adapter.recordings(subj, emotion)
        for rec in recs[:4]:  # не более 4 записей на актора (скорость)
            try:
                fv = pipeline.run(rec.path)
                if fv.combined is not None and len(fv.combined) > 0:
                    vectors.append(fv.combined)
            except Exception:
                pass
        progress.progress((i + 1) / len(subjects), text=f"Обработан {subj}")
    progress.empty()
    return np.stack(vectors).astype(np.float32) if vectors else None


def render(repo: ProfileRepo, config: AppConfig) -> None:
    st.title("⚙️ Обучение нормализатора")
    st.markdown(
        "Нормализатор масштабирует вектор признаков в `[0, 1]` (требование ГОСТ) "
        "и сжимает его через PCA перед подачей в НПБК."
    )

    # Если нормализатор уже обучен — показать инфо
    st.markdown("### Существующие нормализаторы")
    try:
        all_norms = repo.list_normalizers()
        if all_norms:
            for n in all_norms:
                st.info(
                    f"**#{n.id}** — эмоция: `{n.emotion}` | "
                    f"векторов: {n.n_vectors} | "
                    f"вход: {n.input_dim}D → выход: {n.output_dim}D | "
                    f"создан: {n.created_at:%Y-%m-%d %H:%M}"
                )
        else:
            st.caption("Нормализаторов пока нет. Обучи ниже.")
    except Exception as e:
        st.warning(f"Не удалось загрузить список нормализаторов: {e}")

    st.markdown("---")
    st.markdown("### Обучить новый нормализатор")

    adapters = _get_adapters()
    if not adapters:
        st.error(f"Датасеты не найдены в `{DATASET_ROOT}`.")
        return

    col_ds, col_emo = st.columns(2)
    with col_ds:
        ds_name = st.selectbox("Датасет", list(adapters.keys()), key="norm_ds")
    adapter = adapters[ds_name]

    with col_emo:
        emotion = st.selectbox("Эмоция", adapter.emotions(), key="norm_emo")

    n_actors = st.slider(
        "Количество случайных акторов",
        min_value=5,
        max_value=min(24, len(adapter.subjects())),
        value=min(12, len(adapter.subjects())),
        key="norm_n_actors",
        help="Больше акторов → лучше нормализация, дольше обучение",
    )
    pca_dim = st.slider(
        "Размерность после PCA",
        min_value=32,
        max_value=512,
        value=256,
        step=32,
        key="norm_pca_dim",
        help="256 — рекомендовано. Можно определить оптимум через experiment_pca_dim()",
    )

    if st.button("🚀 Обучить нормализатор", key="norm_train_btn"):
        subjects = adapter.subjects()
        random.shuffle(subjects)
        selected = subjects[:n_actors]

        st.info(f"Собираю векторы от {n_actors} акторов ({emotion})...")
        vectors = _collect_vectors(adapter, selected, emotion)

        if vectors is None or len(vectors) < 10:
            st.error(
                f"Не удалось собрать достаточно векторов (нужно ≥ 10, получено "
                f"{len(vectors) if vectors is not None else 0}). "
                "Проверь путь к датасету и выбранную эмоцию."
            )
            return

        input_dim = vectors.shape[1]
        actual_pca_dim = min(pca_dim, input_dim, len(vectors) - 1)

        with st.spinner("Обучаю MinMax + PCA..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(vectors)
            pca = PCA(n_components=actual_pca_dim, random_state=42)
            pca.fit(scaled)

        explained = pca.explained_variance_ratio_.sum()
        st.success(
            f"Готово! Обучено на {len(vectors)} векторах. "
            f"PCA: {input_dim}D → {actual_pca_dim}D "
            f"(объяснённая дисперсия: {explained:.1%})"
        )

        # Сохраняем в БД
        import pickle
        normalizer_obj = {"scaler": scaler, "pca": pca}
        nid = repo.save_normalizer(
            emotion=emotion,
            normalizer_obj=normalizer_obj,
            n_vectors=len(vectors),
            input_dim=input_dim,
            output_dim=actual_pca_dim,
        )
        st.success(f"Нормализатор сохранён в БД с id={nid}")

        # Показываем сводку
        col1, col2, col3 = st.columns(3)
        col1.metric("Векторов", len(vectors))
        col2.metric("Входная размерность", input_dim)
        col3.metric("Выходная размерность", actual_pca_dim)

        st.info(
            f"Для регистрации с эмоцией **{emotion}** будет автоматически "
            f"применён нормализатор **#{nid}**."
        )
