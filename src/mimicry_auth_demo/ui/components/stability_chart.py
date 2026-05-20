"""Bar chart of per-neuron stability indices ω_i."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def render_stability_chart(
    stability_indices: np.ndarray,
    threshold: float = 0.6,
    title: str = "Neuron Stability (ω_i)",
) -> None:
    if len(stability_indices) == 0:
        st.info("No stability data available.")
        return

    colors = ["#e74c3c" if w < threshold else "#2ecc71" for w in stability_indices]
    fig, ax = plt.subplots(figsize=(max(6, len(stability_indices) / 5), 3))
    ax.bar(range(len(stability_indices)), stability_indices, color=colors, width=0.8)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"threshold={threshold}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("ω_i")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
