"""Pipeline stage output visualization."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mimicry_preproc.pipeline import StageOutput
from mimicry_preproc.stages.landmark_extractor import W300_TO_MP


STAGE_NAMES = [
    "video_reader",
    "face_detector",
    "quality_filter",
    "stabilizer",
    "landmark_extractor",
    "feature_extractor",
]


def render_stage_selector() -> str:
    return st.selectbox("Pipeline stage", STAGE_NAMES, index=4)


def render_stage_output(stage: str, out: StageOutput) -> None:
    st.subheader(f"Stage: {stage}")

    timing = out.timings_ms.get(stage)
    if timing is not None:
        st.caption(f"⏱ {timing:.1f} ms")

    if stage == "video_reader":
        _render_frames(out.frames or [])
    elif stage == "face_detector":
        _render_faces(out.faces or [])
    elif stage == "quality_filter":
        _render_faces(out.faces_filtered or [])
    elif stage == "stabilizer":
        _render_faces(out.faces_stabilized or [])
    elif stage == "landmark_extractor":
        _render_landmarks(out.landmarks or [], out.faces_stabilized or [])
    elif stage == "feature_extractor":
        _render_feature_vector(out.feature_vector)


def _render_frames(frames: list) -> None:
    if not frames:
        st.warning("No frames")
        return
    cols = st.columns(min(5, len(frames)))
    for col, frame in zip(cols, frames[:5]):
        img = frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        col.image(img, caption=f"t={frame.timestamp_ms:.0f}ms", use_container_width=True)


def _render_faces(faces: list) -> None:
    valid = [f for f in faces if f is not None]
    if not valid:
        st.warning("No faces detected")
        return
    cols = st.columns(min(5, len(valid)))
    for col, face in zip(cols, valid[:5]):
        img = face.aligned if face.aligned is not None else face.frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        col.image(img, caption=f"conf={face.confidence:.2f}", use_container_width=True)


def _render_landmarks(landmarks: list, faces: list) -> None:
    if not landmarks:
        st.warning("No landmarks detected")
        return
    lm = landmarks[len(landmarks) // 2]  # show middle frame
    face = lm.face_region
    img = face.aligned if face.aligned is not None else face.frame.image
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    overlay = img_rgb.copy()
    for i, (px, py) in enumerate(lm.points):
        x, y = int(px * w), int(py * h)
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)

    st.image(overlay, caption=f"Frame {lm.face_region.frame.index} — 68 landmarks", use_container_width=True)
    st.caption(f"Mean visibility: {lm.visibility.mean():.2f}")


def _render_feature_vector(fv) -> None:
    if fv is None or fv.combined.size == 0:
        st.warning("No feature vector")
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
    parts = [
        ("Geometric", fv.geometric),
        ("Texture", fv.texture),
        ("Motion", fv.motion),
        ("Combined", fv.combined),
    ]
    for ax, (name, arr) in zip(axes, parts):
        if arr.size > 0:
            ax.plot(arr, linewidth=0.5)
        ax.set_title(f"{name}\n({len(arr)})", fontsize=8)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
