"""
Real-time video processor for WebRTC stream.

Processes each incoming frame through selected pipeline stages and
draws the result as an overlay. Works inside the streamlit-webrtc
recv() callback (runs in a background thread).

Supported overlay modes:
  "raw"              — no processing, passthrough
  "face_detector"    — bounding box + confidence
  "quality_filter"   — bbox + sharpness/brightness scores
  "landmark_extractor" — 68-point face mesh (300W standard subset)
  "all_landmarks"    — all 478 MediaPipe FaceMesh points
  "feature_extractor"  — landmark overlay + feature vector stats bar
  "stabilizer"       — optical flow arrows
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
    "stabilizer", "landmark_extractor", "all_landmarks", "feature_extractor",
]

_ALL_STAGES: list[StageMode] = [
    "raw", "face_detector", "quality_filter",
    "stabilizer", "landmark_extractor", "all_landmarks", "feature_extractor",
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
    The Streamlit main thread writes to `mode`;
    recv() reads it under a lock.

    Jitter reduction:
    - MediaPipe FaceDetector and FaceLandmarker run in VIDEO mode so they
      use internal temporal smoothing between frames.
    - EMA (alpha=0.35) is applied to bbox coordinates and landmark points
      for additional stability on top of MediaPipe's own filtering.
    """

    # EMA smoothing factor: lower = smoother but more lag, higher = less lag but more jitter
    _EMA_ALPHA: float = 0.35

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode: StageMode = "landmark_extractor"
        self._face_detector_mp = None    # raw MediaPipe VIDEO-mode detector
        self._landmarker_mp = None       # raw MediaPipe VIDEO-mode landmarker
        self._quality_filter = None
        self._prev_gray: np.ndarray | None = None
        self._fps_t: float = time.time()
        self._fps: float = 0.0
        self._timestamp_ms: int = 0

        # EMA state for smoothing
        self._smooth_bbox: np.ndarray | None = None   # (4,) float [x,y,w,h]
        self._smooth_pts68: np.ndarray | None = None  # (68, 2) float
        self._smooth_pts478: np.ndarray | None = None  # (478, 2) float

        self._init_pipeline()

    # ------------------------------------------------------------------
    # Init — create VIDEO-mode MediaPipe objects directly for low latency
    # ------------------------------------------------------------------
    def _init_pipeline(self) -> None:
        try:
            import mediapipe as mp
            from pathlib import Path

            model_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "mimicry-preproc" / "models"

            BaseOptions = mp.tasks.BaseOptions
            RunningMode = mp.tasks.vision.RunningMode

            # Face detector in VIDEO mode
            det_opts = mp.tasks.vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_dir / "face_detector.tflite")),
                running_mode=RunningMode.VIDEO,
                min_detection_confidence=0.4,
            )
            self._face_detector_mp = mp.tasks.vision.FaceDetector.create_from_options(det_opts)

            # Face landmarker in VIDEO mode — enables MediaPipe temporal filtering
            lm_opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker_mp = mp.tasks.vision.FaceLandmarker.create_from_options(lm_opts)

            from mimicry_preproc.stages.quality_filter import QualityFilter, QualityFilterConfig
            self._quality_filter = QualityFilter(QualityFilterConfig(min_sharpness=30.0))

        except Exception as e:
            print(f"[RealtimeProcessor] init error: {e}")

    def _to_mp_image(self, img_bgr: np.ndarray):
        import mediapipe as mp
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        )

    def _ema(self, new: np.ndarray, old: np.ndarray | None) -> np.ndarray:
        if old is None:
            return new.astype(np.float32)
        return (self._EMA_ALPHA * new + (1 - self._EMA_ALPHA) * old).astype(np.float32)

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

        # Increment timestamp for MediaPipe VIDEO mode
        self._timestamp_ms += 33  # ~30 FPS

        try:
            out = self._process(img_bgr, mode)
        except Exception:
            out = img_bgr.copy()

        # FPS counter
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._fps_t, 1e-4))
        self._fps_t = now
        cv2.putText(out, f"{self._fps:.0f} FPS", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GRAY, 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    # ------------------------------------------------------------------
    # Processing per mode — uses VIDEO-mode MediaPipe + EMA smoothing
    # ------------------------------------------------------------------
    def _process(self, img: np.ndarray, mode: StageMode) -> np.ndarray:
        if mode == "raw":
            return img

        mp_img = self._to_mp_image(img)
        ih, iw = img.shape[:2]

        # --- Face detection (VIDEO mode → temporal smoothing) ---
        raw_bbox: np.ndarray | None = None
        if self._face_detector_mp is not None:
            det_result = self._face_detector_mp.detect_for_video(mp_img, self._timestamp_ms)
            if det_result.detections:
                d = det_result.detections[0]
                bb = d.bounding_box
                raw_bbox = np.array([bb.origin_x, bb.origin_y, bb.width, bb.height], np.float32)

        # EMA on bbox
        if raw_bbox is not None:
            self._smooth_bbox = self._ema(raw_bbox, self._smooth_bbox)
        bbox_smooth = self._smooth_bbox

        if mode == "face_detector":
            return self._draw_face_smooth(img, bbox_smooth,
                                          det_result.detections[0].categories[0].score
                                          if (raw_bbox is not None and
                                              self._face_detector_mp and
                                              det_result.detections) else 0.0)

        if mode == "quality_filter":
            return self._draw_quality_smooth(img, bbox_smooth)

        if bbox_smooth is None:
            _draw_text(img.copy(), "No face detected", (10, 50), _RED)
            return img

        if mode == "stabilizer":
            return self._draw_stabilizer_raw(img, bbox_smooth)

        # --- Landmarks (VIDEO mode → temporal smoothing) ---
        raw_pts68: np.ndarray | None = None
        raw_pts478: np.ndarray | None = None
        if self._landmarker_mp is not None:
            lm_result = self._landmarker_mp.detect_for_video(mp_img, self._timestamp_ms)
            if lm_result.face_landmarks:
                mp_lm = lm_result.face_landmarks[0]
                # All 478 points
                raw_pts478 = np.array([[lm.x, lm.y] for lm in mp_lm], np.float32)
                # 68-point subset via W300_TO_MP mapping
                from mimicry_preproc.stages.landmark_extractor import W300_TO_MP
                pts68 = np.zeros((68, 2), np.float32)
                for w300_idx in range(68):
                    mp_idx = W300_TO_MP.get(w300_idx)
                    if mp_idx is not None and mp_idx < len(mp_lm):
                        pts68[w300_idx] = [mp_lm[mp_idx].x, mp_lm[mp_idx].y]
                raw_pts68 = pts68

        # EMA on landmark points
        if raw_pts68 is not None:
            self._smooth_pts68 = self._ema(raw_pts68, self._smooth_pts68)
        if raw_pts478 is not None:
            self._smooth_pts478 = self._ema(raw_pts478, self._smooth_pts478)

        if mode == "landmark_extractor":
            return self._draw_landmarks_smooth(img, bbox_smooth, self._smooth_pts68)

        if mode == "all_landmarks":
            return self._draw_all_landmarks_smooth(img, bbox_smooth, self._smooth_pts478)

        if mode == "feature_extractor":
            return self._draw_features_smooth(img, bbox_smooth, self._smooth_pts68)

        return img

    # ------------------------------------------------------------------
    # Drawing helpers (all use smoothed coordinates)
    # ------------------------------------------------------------------
    def _draw_face_smooth(self, img: np.ndarray, bbox: np.ndarray | None, conf: float) -> np.ndarray:
        out = img.copy()
        if bbox is None:
            _draw_text(out, "No face", (10, 50), _RED)
            return out
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(out, (x, y), (x + w, y + h), _GREEN, 2)
        _draw_text(out, f"conf={conf:.2f}", (x, max(y - 8, 12)), _GREEN)
        return out

    def _draw_quality_smooth(self, img: np.ndarray, bbox: np.ndarray | None) -> np.ndarray:
        out = img.copy()
        if bbox is None:
            _draw_text(out, "No face", (10, 50), _RED)
            return out
        x, y, w, h = bbox.astype(int)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        crop = gray[max(0, y):y2, max(0, x):x2]
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 0 else 0.0
        brightness = float(crop.mean()) / 255.0 if crop.size > 0 else 0.0
        passed = sharpness >= 30.0 and 0.1 <= brightness <= 0.95
        color = _GREEN if passed else _RED
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        for i, txt in enumerate([f"sharp={sharpness:.0f}", f"bright={brightness:.2f}",
                                   "PASS" if passed else "FAIL"]):
            _draw_text(out, txt, (x, max(y - 8 - i * 18, 12)), color)
        return out

    def _draw_stabilizer_raw(self, img: np.ndarray, bbox: np.ndarray | None) -> np.ndarray:
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if bbox is not None:
            x, y, w, h = bbox.astype(int)
            cv2.rectangle(out, (x, y), (x + w, y + h), _GREEN, 1)
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                pts = np.array([[x + w * fx, y + h * fy]
                                 for fx in [0.25, 0.5, 0.75]
                                 for fy in [0.25, 0.5, 0.75]], np.float32).reshape(-1, 1, 2)
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

    def _draw_landmarks_smooth(self, img: np.ndarray, bbox: np.ndarray | None,
                                pts68: np.ndarray | None) -> np.ndarray:
        out = img.copy()
        if bbox is not None:
            x, y, w, h = bbox.astype(int)
            cv2.rectangle(out, (x, y), (x + w, y + h), _GRAY, 1)

        if pts68 is None:
            _draw_text(out, "No landmarks", (10, 50), _RED)
            return out

        ih, iw = img.shape[:2]
        pts_px = (pts68 * np.array([iw, ih])).astype(int)

        for i, j in _LANDMARK_CONNECTIONS:
            if i < len(pts_px) and j < len(pts_px):
                cv2.line(out, tuple(pts_px[i]), tuple(pts_px[j]), _GRAY, 1, cv2.LINE_AA)

        for px, py in pts_px:
            cv2.circle(out, (int(px), int(py)), 2, _GREEN, -1, cv2.LINE_AA)

        _draw_text(out, "68 pts (300W) — EMA smoothed", (10, 50), _GREEN)
        return out

    def _draw_all_landmarks_smooth(self, img: np.ndarray, bbox: np.ndarray | None,
                                    pts478: np.ndarray | None) -> np.ndarray:
        out = img.copy()
        if bbox is not None:
            x, y, w, h = bbox.astype(int)
            cv2.rectangle(out, (x, y), (x + w, y + h), _GRAY, 1)

        if pts478 is None:
            _draw_text(out, "No landmarks", (10, 50), _RED)
            return out

        ih, iw = img.shape[:2]
        for i, (nx, ny) in enumerate(pts478):
            px, py = int(nx * iw), int(ny * ih)
            cv2.circle(out, (px, py), 1, (0, 200, 80), -1, cv2.LINE_AA)

        _draw_text(out, f"478 pts (MediaPipe) — EMA smoothed", (10, 50), _GREEN)
        return out

    def _draw_all_landmarks(self, img: np.ndarray, face_region) -> np.ndarray:
        """Legacy — redirects to smooth version."""
        return self._draw_all_landmarks_smooth(img, None, self._smooth_pts478)

    def _draw_features_smooth(self, img: np.ndarray, bbox: np.ndarray | None,
                               pts68: np.ndarray | None) -> np.ndarray:
        out = self._draw_landmarks_smooth(img, bbox, pts68)

        if pts68 is not None:
            from mimicry_preproc.features.geometric import extract_from_frame
            geo = extract_from_frame(pts68)
            n = min(30, len(geo))
            _mini_bar_chart(out, geo[:n], 4, out.shape[0] - 64, 3, 60, _GREEN, _RED)
            _draw_text(out, f"geo[:{n}]", (4, out.shape[0] - 68), _GRAY, scale=0.4)

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
        _ALL_STAGES,
        index=_ALL_STAGES.index(mode) if mode in _ALL_STAGES else 4,
        key=f"_rt_mode_{key}",
        format_func=lambda s: {
            "raw": "raw — без обработки",
            "face_detector": "face_detector — bbox лица",
            "quality_filter": "quality_filter — резкость/яркость",
            "stabilizer": "stabilizer — optical flow",
            "landmark_extractor": "landmark_extractor — 68 точек 300W",
            "all_landmarks": "all_landmarks — все 478 точек MediaPipe",
            "feature_extractor": "feature_extractor — 68 точек + гистограмма",
        }.get(s, s),
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
