"""
Live pipeline settings editor with per-stage parameter controls.
Settings are persisted in Streamlit session_state and survive stage switching.
"""
from __future__ import annotations

import streamlit as st

from mimicry_preproc.pipeline import PipelineConfig
from mimicry_preproc.stages.face_detector import FaceDetectorConfig
from mimicry_preproc.stages.feature_extractor import FeatureExtractorConfig
from mimicry_preproc.stages.landmark_extractor import LandmarkExtractorConfig
from mimicry_preproc.stages.normalizer import NormalizerConfig
from mimicry_preproc.stages.quality_filter import QualityFilterConfig
from mimicry_preproc.stages.stabilizer import StabilizerConfig
from mimicry_preproc.stages.video_reader import VideoReaderConfig


def _state_key(stage: str, param: str) -> str:
    return f"pipeline_cfg_{stage}_{param}"


def get_current_config() -> PipelineConfig:
    """Build PipelineConfig from current session_state values."""
    s = st.session_state

    def get(stage: str, param: str, default):
        return s.get(_state_key(stage, param), default)

    return PipelineConfig(
        video_reader=VideoReaderConfig(
            target_fps=get("video_reader", "target_fps", 25.0),
            max_duration_sec=get("video_reader", "max_duration_sec", 10.0),
            grayscale=get("video_reader", "grayscale", True),
        ),
        face_detector=FaceDetectorConfig(
            min_confidence=get("face_detector", "min_confidence", 0.5),
        ),
        quality_filter=QualityFilterConfig(
            min_sharpness=get("quality_filter", "min_sharpness", 50.0),
            min_brightness=get("quality_filter", "min_brightness", 0.1),
            max_brightness=get("quality_filter", "max_brightness", 0.95),
        ),
        stabilizer=StabilizerConfig(
            method=get("stabilizer", "method", "affine"),
        ),
        landmark_extractor=LandmarkExtractorConfig(
            min_confidence=get("landmark_extractor", "min_confidence", 0.5),
        ),
        feature_extractor=FeatureExtractorConfig(
            use_geometric=get("feature_extractor", "use_geometric", True),
            use_texture=get("feature_extractor", "use_texture", True),
            use_motion=get("feature_extractor", "use_motion", True),
            use_deep=get("feature_extractor", "use_deep", False),
        ),
        normalizer=NormalizerConfig(
            reduce=get("normalizer", "reduce", "pca"),
        ),
    )


def render_stage_settings(stage: str) -> bool:
    """
    Render editable settings for the selected stage.
    Returns True if any value was changed (triggers rerun).
    """
    changed = False

    def slider(stage_: str, param: str, label: str, min_val, max_val, default, step=None):
        nonlocal changed
        key = _state_key(stage_, param)
        prev = st.session_state.get(key, default)
        kw = {"step": step} if step is not None else {}
        val = st.slider(label, min_val, max_val, prev, key=key, **kw)
        if val != prev:
            changed = True
        return val

    def toggle(stage_: str, param: str, label: str, default: bool):
        nonlocal changed
        key = _state_key(stage_, param)
        prev = st.session_state.get(key, default)
        val = st.checkbox(label, prev, key=key)
        if val != prev:
            changed = True
        return val

    def selectbox(stage_: str, param: str, label: str, options: list, default):
        nonlocal changed
        key = _state_key(stage_, param)
        prev = st.session_state.get(key, default)
        idx = options.index(prev) if prev in options else 0
        val = st.selectbox(label, options, index=idx, key=key)
        if val != prev:
            changed = True
        return val

    if stage == "video_reader":
        slider("video_reader", "target_fps", "Target FPS", 5.0, 60.0, 25.0, 1.0)
        slider("video_reader", "max_duration_sec", "Max duration (sec)", 1.0, 30.0, 10.0, 0.5)
        toggle("video_reader", "grayscale", "Grayscale", True)

    elif stage == "face_detector":
        slider("face_detector", "min_confidence", "Min detection confidence", 0.1, 1.0, 0.5, 0.05)

    elif stage == "quality_filter":
        slider("quality_filter", "min_sharpness", "Min sharpness (Laplacian var)", 0.0, 500.0, 50.0, 5.0)
        slider("quality_filter", "min_brightness", "Min brightness", 0.0, 0.5, 0.1, 0.01)
        slider("quality_filter", "max_brightness", "Max brightness", 0.5, 1.0, 0.95, 0.01)

    elif stage == "stabilizer":
        selectbox("stabilizer", "method", "Stabilization method", ["affine", "optical_flow"], "affine")

    elif stage == "landmark_extractor":
        slider("landmark_extractor", "min_confidence", "Min tracking confidence", 0.1, 1.0, 0.5, 0.05)

    elif stage == "feature_extractor":
        toggle("feature_extractor", "use_geometric", "Geometric features", True)
        toggle("feature_extractor", "use_texture", "Texture features (LBP+HOG+Gabor)", True)
        toggle("feature_extractor", "use_motion", "Motion features (optical flow)", True)
        toggle("feature_extractor", "use_deep", "Deep embeddings (requires torch)", False)

    elif stage == "normalizer":
        selectbox("normalizer", "reduce", "Dimensionality reduction", ["pca", "lda", "none"], "pca")

    return changed


def export_config_button(config: PipelineConfig) -> None:
    import json
    data = json.dumps(config.to_dict(), indent=2, default=str)
    st.download_button(
        "Export pipeline_config.json",
        data=data,
        file_name="pipeline_config.json",
        mime="application/json",
    )
