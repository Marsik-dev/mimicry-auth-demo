"""Visualize per-layer FAR/FRR and stability metrics from TrainingResult."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from npbk.types import LayerReport, TrainingResult


def render_layer_reports(result: TrainingResult) -> None:
    """Show Layer 1 and Layer 2 quality side by side after training."""
    r1, r2 = result.layer1_report, result.layer2_report
    if r1 is None or r2 is None:
        return

    st.markdown("### Качество по слоям НПБК")
    st.caption(
        "Ошибки первого рода (FRR) — отказ своему. "
        "Ошибки второго рода (FAR) — пропуск чужого."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Слой 1** — первичное кодирование")
        _render_single(r1)

    with col2:
        st.markdown("**Слой 2** — коррекция ошибок")
        _render_single(r2)

    # Improvement delta
    st.markdown("---")
    st.markdown("**Улучшение после Слоя 2:**")
    c1, c2, c3 = st.columns(3)
    delta_far = r1.empirical_far - r2.empirical_far
    delta_frr = r1.empirical_frr - r2.empirical_frr
    delta_stab = r2.mean_stability_own - r1.mean_stability_own
    c1.metric("ΔFAR (↓ лучше)", f"{delta_far:+.3f}",
              delta=f"{delta_far:.3f}", delta_color="normal")
    c2.metric("ΔFRR (↓ лучше)", f"{delta_frr:+.3f}",
              delta=f"{delta_frr:.3f}", delta_color="normal")
    c3.metric("ΔСтабильность (↑ лучше)", f"{delta_stab:+.3f}",
              delta=f"{delta_stab:.3f}", delta_color="normal")


def _render_single(r: LayerReport) -> None:
    col_a, col_b = st.columns(2)
    col_a.metric("FAR (предсказан)", f"{r.predicted_far:.4f}")
    col_b.metric("FAR (эмпирический)", f"{r.empirical_far:.4f}",
                 delta=f"{r.empirical_far - r.predicted_far:+.4f}", delta_color="off")
    col_a.metric("FRR (предсказан)", f"{r.predicted_frr:.4f}")
    col_b.metric("FRR (эмпирический)", f"{r.empirical_frr:.4f}",
                 delta=f"{r.empirical_frr - r.predicted_frr:+.4f}", delta_color="off")
    col_a.metric("Нейронов выбрано", f"{r.n_neurons_selected} / {r.n_neurons_total}")
    col_b.metric("Средняя стабильность ω", f"{r.mean_stability_own:.3f}")

    # Stability histogram
    if r.n_neurons_selected > 0:
        fig, ax = plt.subplots(figsize=(4, 1.8))
        n = r.n_neurons_selected
        colors = ["#e74c3c" if w < 0.6 else "#2ecc71" for w in r.stability_own]
        ax.bar(range(n), r.stability_own, color=colors, width=0.8)
        ax.axhline(0.6, color="orange", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Нейрон", fontsize=7)
        ax.set_ylabel("ω_i", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig)
        plt.close(fig)
