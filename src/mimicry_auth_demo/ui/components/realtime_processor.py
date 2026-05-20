"""
Real-time video processor for WebRTC stream.

Processes each incoming frame through selected pipeline stages and
draws the result as an overlay. Works inside the streamlit-webrtc
recv() callback (runs in a background thread).

Supported overlay modes:
  "raw"              — no processing, passthrough
  "face_detector"    — bounding box + confidence
  "quality_filter"   — bbox + sharpness/brightness scores
  "landmark_extractor" — 68-point face mesh
  "feature_extractor"  — landmark overlay + feature vector stats bar
  "stabilizer"       — landmark overlay on stabilized frame
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

_RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

StageMode = Literal[
    "raw", "face_detector", "quality_filter",
    "stabilizer", "landmark_extractor", "feature_extractor",
]

_LANDMARK_CONNECTIONS = [
    # Jawline
    *zip(range(0, 16), range(1, 17)),
    # Right eyebrow
    *zip(range(17, 21), range(18, 22)),
    # Left eyebrow
    *zip(range(22, 26), range(23, 27)),
    # Nose bridge
    *zip(range(27, 30), range(28, 31)),
    # Nose base
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Right eye
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Left eye
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer mouth
    *zip(range(48, 59), range(49, 60)), (59, 48),
    # Inner mouth
    *zip(range(60, 67), range(61, 68)), (67, 60),
]

# Colors (BGR)
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 220)
_YELLOW = (0, 200, 220)
_WHITE  = (240, 240, 240)
_GRAY   = (120, 120, 120)


class _PipelineVideoProcessor(VideoProcessorBase):
    """
    Stateful per-session video processor.
    The Streamlit main thread writes to `mode` and `config`;
    recv() reads them under a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode: StageMode = "landmark_extractor"
        self._face_detector = None
        self._landmark_extractor = None
        self._quality_filter = None
        self._prev_gray: np.ndarray | None = None
        self._fps_t: float = time.time()
        self._fps: float = 0.0
        self._init_pipeline()

    # ------------------------------------------------------------------
    # Lazy init (runs in recv thread)
    # ------------------------------------------------------------------
    def _init_pipeline(self) -> None:
        try:
            from mimicry_preproc.stages.face_detector import FaceDetector, FaceDetectorConfig
            from mimicry_preproc.stages.landmark_extractor import LandmarkExtractor, LandmarkExtractorConfig
            from mimicry_preproc.stages.quality_filter import QualityFilter, QualityFilterConfig

            self._face_detector = FaceDetector(FaceDetectorConfig(min_confidence=0.4))
            self._landmark_extractor = LandmarkExtractor(LandmarkExtractorConfig(min_confidence=0.4))
            self._quality_filter = QualityFilter(QualityFilterConfig(min_sharpness=30.0))
        except Exception as e:
            print(f"[RealtimeProcessor] init error: {e}")

    # ------------------------------------------------------------------
    # Public API — called from Streamlit main thread
    # ------------------------------------------------------------------
    @property
    def mode(self) -> StageMode:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: StageMode) -> None:
        with self._lock:
            self._mode = value

    # ------------------------------------------------------------------
    # recv — called per frame in background thread
    # ------------------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        with self._lock:
            mode = self._mode

        try:
            out = self._process(img_bgr, mode)
        except Exception:
            out = img_bgr

        # FPS counter
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._fps_t, 1e-4))
        self._fps_t = now
        cv2.putText(out, f"{self._fps:.0f} FPS", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GRAY, 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    # ------------------------------------------------------------------
    # Processing per mode
    # ------------------------------------------------------------------
    def _process(self, img: np.ndarray, mode: StageMode) -> np.ndarray:
        if mode == "raw":
            return img

        from mimicry_preproc.types import Frame
        frame = Frame(image=img, timestamp_ms=0.0, index=0)

        # Face detection (needed for all modes except raw)
        face_region = None
        if self._face_detector is not None:
            face_region = self._face_detector.detect(frame)

        if mode == "face_detector":
            return self._draw_face(img, face_region)

        if mode == "quality_filter":
            return self._draw_quality(img, face_region)

        if face_region is None:
            _draw_text(img, "No face detected", (10, 50), _RED)
            return img

        if mode == "stabilizer":
            return self._draw_stabilizer(img, face_region)

        # Landmark extraction (needed for landmark and feature modes)
        landmarks = None
        if self._landmark_extractor is not None:
            landmarks = self._landmark_extractor.extract(face_region)

        if mode == "landmark_extractor":
            return self._draw_landmarks(img, face_region, landmarks)

        if mode == "feature_extractor":
            return self._draw_features(img, face_region, landmarks)

        return img

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_face(self, img: np.ndarray, face_region) -> np.ndarray:
        out = img.copy()
        if face_region is None:
            _draw_text(out, "No face", (10, 50), _RED)
            return out
        x, y, w, h = face_region.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), _GREEN, 2)
        _draw_text(out, f"conf={face_region.confidence:.2f}", (x, y - 8), _GREEN)
        return out

    def _draw_quality(self, img: np.ndarray, face_region) -> np.ndarray:
        out = img.copy()
        if face_region is None:
            _draw_text(out, "No face", (10, 50), _RED)
            return out

        score = self._quality_filter.assess(face_region) if self._quality_filter else None
        x, y, w, h = face_region.bbox
        color = _GREEN if (score and score.passed) else _RED
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        if score:
            lines = [
                f"sharp={score.sharpness:.0f}",
                f"bright={score.brightness:.2f}",
                f"{'PASS' if score.passed else 'FAIL'}",
            ]
            for i, line in enumerate(lines):
                _draw_text(out, line, (x, y - 8 - i * 18), color)
        return out

    def _draw_stabilizer(self, img: np.ndarray, face_region) -> np.ndarray:
        out = img.copy()
        x, y, w, h = face_region.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), _GREEN, 2)

        # Show optical flow arrows if we have a previous frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            # Compute sparse optical flow on grid points inside face bbox
            pts = np.array([
                [x + w * fx, y + h * fy]
                for fx in [0.25, 0.5, 0.75]
                for fy in [0.25, 0.5, 0.75]
            ], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            for (px, py), (nx, ny), ok in zip(
                pts.reshape(-1, 2), next_pts.reshape(-1, 2), status.reshape(-1)
            ):
                if ok:
                    cv2.arrowedLine(out, (int(px), int(py)), (int(nx), int(ny)),
                                    _YELLOW, 2, tipLength=0.4)
        self._prev_gray = gray
        _draw_text(out, "Stabilizer / Optical Flow", (10, 50), _YELLOW)
        return out

    def _draw_landmarks(self, img: np.ndarray, face_region, landmarks) -> np.ndarray:
        out = img.copy()
        x, y, w, h = face_region.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), _GRAY, 1)

        if landmarks is None:
            _draw_text(out, "Landmarks: none", (10, 50), _RED)
            return out

        ih, iw = img.shape[:2]
        pts_px = np.column_stack([
            (landmarks.points[:, 0] * iw).astype(int),
            (landmarks.points[:, 1] * ih).astype(int),
        ])

        # Connections
        for i, j in _LANDMARK_CONNECTIONS:
            if i < len(pts_px) and j < len(pts_px):
                cv2.line(out, tuple(pts_px[i]), tuple(pts_px[j]), _GRAY, 1, cv2.LINE_AA)

        # Points (color by visibility)
        for k, (px, py) in enumerate(pts_px):
            vis = float(landmarks.visibility[k])
            color = (0, int(80 + 140 * vis), int(200 * vis))
            cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

        vis_mean = float(landmarks.visibility.mean())
        _draw_text(out, f"68 pts  vis={vis_mean:.2f}", (10, 50), _GREEN)
        return out

    def _draw_features(self, img: np.ndarray, face_region, landmarks) -> np.ndarray:
        # Draw landmarks first
        out = self._draw_landmarks(img, face_region, landmarks)

        if landmarks is not None:
            from mimicry_preproc.features.geometric import extract_from_frame
            geo = extract_from_frame(landmarks.points)

            # Mini bar chart in bottom-left corner
            n = min(30, len(geo))
            bar_h, bar_w = 60, 3
            pad = 4
            x0, y0 = pad, out.shape[0] - 60 - pad
            _mini_bar_chart(out, geo[:n], x0, y0, bar_w, 60, _GREEN, _RED)
            _draw_text(out, f"geo[:{n}]", (x0, y0 - 5), _GRAY, scale=0.4)

        return out


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

def realtime_pipeline_widget(
    key: str = "realtime",
    initial_mode: StageMode = "landmark_extractor",
) -> tuple[StageMode, "_PipelineVideoProcessor | None"]:
    """
    Render a live WebRTC camera feed with real-time pipeline stage overlay.

    Returns (current_mode, processor) so the caller can read additional state
    (e.g. latest processed frames for enrollment).
    """
    processor_key = f"_rt_proc_{key}"

    mode = st.session_state.get(f"_rt_mode_{key}", initial_mode)

    # Mode selector
    mode = st.selectbox(
        "Стадия пайплайна (live overlay)",
        ["raw", "face_detector", "quality_filter",
         "stabilizer", "landmark_extractor", "feature_extractor"],
        index=["raw", "face_detector", "quality_filter",
               "stabilizer", "landmark_extractor", "feature_extractor"].index(mode),
        key=f"_rt_mode_{key}",
    )

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_RTC_CONFIG,
        video_processor_factory=_PipelineVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    processor: _PipelineVideoProcessor | None = None
    if ctx.video_processor is not None:
        processor = ctx.video_processor
        processor.mode = mode

        if ctx.state.playing:
            st.caption(f"🔴 Live — режим: **{mode}**")
        else:
            st.info("Нажми **START** для запуска камеры.")
    else:
        st.info("Нажми **START** для запуска камеры.")

    return mode, processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _mini_bar_chart(
    img: np.ndarray,
    values: np.ndarray,
    x0: int, y0: int,
    bar_w: int, max_h: int,
    pos_color: tuple, neg_color: tuple,
) -> None:
    vmin, vmax = float(values.min()), float(values.max())
    rng = max(vmax - vmin, 1e-6)
    for i, v in enumerate(values):
        h = int((v - vmin) / rng * max_h)
        color = pos_color if v >= 0 else neg_color
        x = x0 + i * (bar_w + 1)
        cv2.rectangle(img, (x, y0 + max_h - h), (x + bar_w, y0 + max_h), color, -1)
