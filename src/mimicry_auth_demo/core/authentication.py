from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimicry_preproc import FeatureVector, Pipeline
from mimicry_preproc.pipeline import StageOutput
from npbk import NBKContainer, NPBKAuthenticator
from npbk.types import AuthResult

from ..config import AppConfig


@dataclass
class AuthAttemptResult:
    accepted: bool
    auth_result: AuthResult
    feature_vector: FeatureVector
    stage_output: StageOutput
    pipeline_timings_ms: dict[str, float]


class AuthenticationSession:
    def __init__(self, container: NBKContainer, config: AppConfig) -> None:
        self._container = container
        self._config = config
        self._pipeline = Pipeline(config.pipeline)
        self._authenticator = NPBKAuthenticator(container)

    def authenticate(self, video_path: str) -> AuthAttemptResult:
        stage_out = self._pipeline.run_with_debug(video_path)
        return self._finalize(stage_out)

    def authenticate_frames(self, frames_bgr: list[np.ndarray], fps: float = 25.0) -> AuthAttemptResult:
        fv = self._pipeline.run_from_frames(frames_bgr, fps)
        stage_out = StageOutput(feature_vector=fv)
        return self._finalize(stage_out)

    def _finalize(self, stage_out: StageOutput) -> AuthAttemptResult:
        fv = stage_out.feature_vector
        if fv is None or fv.combined.size == 0:
            from npbk.types import AuthResult
            dummy = AuthResult(accepted=False, hamming_distance=-1, confidence=0.0,
                               code=np.array([], dtype=np.int8),
                               details={"error": "no_features"})
            return AuthAttemptResult(
                accepted=False, auth_result=dummy,
                feature_vector=fv or FeatureVector(  # type: ignore[arg-type]
                    geometric=np.zeros(0, dtype=np.float32),
                    texture=np.zeros(0, dtype=np.float32),
                    deep=np.zeros(0, dtype=np.float32),
                    motion=np.zeros(0, dtype=np.float32),
                    combined=np.zeros(0, dtype=np.float32),
                ),
                stage_output=stage_out,
                pipeline_timings_ms=stage_out.timings_ms,
            )

        auth_result = self._authenticator.authenticate(fv.combined)
        return AuthAttemptResult(
            accepted=auth_result.accepted,
            auth_result=auth_result,
            feature_vector=fv,
            stage_output=stage_out,
            pipeline_timings_ms=stage_out.timings_ms,
        )
