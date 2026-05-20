"""Enrollment session: collect own samples → train NPBK → save container."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mimicry_preproc import FeatureVector, Pipeline
from mimicry_preproc.pipeline import StageOutput
from mimicry_preproc.stages.quality_filter import QualityScore
from npbk import NBKContainer, NPBKTrainer, TrainingSet
from npbk.types import TrainingResult

from ..config import AppConfig


@dataclass
class EnrollmentSampleResult:
    feature_vector: FeatureVector
    stage_output: StageOutput
    quality_score: float
    frame_count: int


@dataclass
class EnrollmentProgress:
    samples_collected: int
    samples_required: int
    sample_quality_scores: list[float]
    is_ready: bool

    @property
    def progress_pct(self) -> float:
        return min(1.0, self.samples_collected / max(1, self.samples_required))


class EnrollmentSession:
    def __init__(self, config: AppConfig, user_id: str) -> None:
        self.config = config
        self.user_id = user_id
        self._pipeline = Pipeline(config.pipeline)
        self._own_vectors: list[np.ndarray] = []
        self._quality_scores: list[float] = []

    def process_video(self, video_path: str) -> EnrollmentSampleResult:
        stage_out = self._pipeline.run_with_debug(video_path)
        fv = stage_out.feature_vector
        quality = float(fv.combined.std()) if fv is not None and fv.combined.size > 0 else 0.0
        return EnrollmentSampleResult(
            feature_vector=fv,  # type: ignore[arg-type]
            stage_output=stage_out,
            quality_score=quality,
            frame_count=len(stage_out.frames or []),
        )

    def process_frames(self, frames_bgr: list[np.ndarray], fps: float = 25.0) -> EnrollmentSampleResult:
        fv = self._pipeline.run_from_frames(frames_bgr, fps)
        quality = float(fv.combined.std()) if fv.combined.size > 0 else 0.0
        return EnrollmentSampleResult(
            feature_vector=fv,
            stage_output=StageOutput(),
            quality_score=quality,
            frame_count=len(frames_bgr),
        )

    def add_sample(self, result: EnrollmentSampleResult) -> None:
        if result.feature_vector is not None and result.feature_vector.combined.size > 0:
            self._own_vectors.append(result.feature_vector.combined)
            self._quality_scores.append(result.quality_score)

    def is_ready_to_train(self) -> bool:
        return len(self._own_vectors) >= self.config.n_required_own_samples

    def get_progress(self) -> EnrollmentProgress:
        return EnrollmentProgress(
            samples_collected=len(self._own_vectors),
            samples_required=self.config.n_required_own_samples,
            sample_quality_scores=list(self._quality_scores),
            is_ready=self.is_ready_to_train(),
        )

    def train(self, other_vectors: np.ndarray) -> tuple[TrainingResult, NBKContainer]:
        if not self.is_ready_to_train():
            raise RuntimeError(
                f"Not enough samples: {len(self._own_vectors)} / {self.config.n_required_own_samples}"
            )
        own = np.stack(self._own_vectors)
        ts = TrainingSet(own_vectors=own, other_vectors=other_vectors)
        result = NPBKTrainer(self.config.trainer).train(ts)
        container = NBKContainer.from_result(
            result,
            feature_dim=own.shape[1],
            metadata={"user_id": self.user_id},
        )
        return result, container
