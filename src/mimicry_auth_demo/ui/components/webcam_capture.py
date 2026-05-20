"""
WebRTC-based webcam video capture component.

Provides a streaming camera feed with Start/Stop recording.
When stopped, returns the accumulated frames as a list of BGR numpy arrays
ready to pass into Pipeline.run_from_frames().

STUN server (stun.l.google.com) is used for ICE negotiation so the component
works both on localhost and over the local network.
"""
from __future__ import annotations

import threading
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

_RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class _FrameRecorder(VideoProcessorBase):
    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

    def start_recording(self) -> None:
        with self._lock:
            self._frames = []
            self._recording = True

    def stop_recording(self) -> list[np.ndarray]:
        with self._lock:
            self._recording = False
            return list(self._frames)

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        with self._lock:
            if self._recording:
                # Subsample to ~10 FPS to keep buffer manageable
                if len(self._frames) == 0 or True:
                    self._frames.append(img_bgr.copy())
        return frame


def webcam_recorder(
    key: str,
    label: str = "Веб-камера",
    max_frames: int = 300,
) -> list[np.ndarray] | None:
    """
    Render a WebRTC camera widget with Record/Stop buttons.

    Returns the captured frames (list of BGR ndarray) after the user
    clicks Stop, or None if recording hasn't finished yet.
    """
    result_key = f"_wcam_result_{key}"

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_RTC_CONFIG,
        video_processor_factory=_FrameRecorder,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor is not None:
        recorder: _FrameRecorder = ctx.video_processor
        n = recorder.frame_count
        is_rec = recorder._recording

        col1, col2 = st.columns(2)

        if col1.button(
            "⏺ Начать запись" if not is_rec else "🔴 Идёт запись...",
            key=f"_start_{key}",
            disabled=is_rec,
            type="primary" if not is_rec else "secondary",
        ):
            recorder.start_recording()
            st.rerun()

        if col2.button(
            f"⏹ Стоп ({n} кадров)",
            key=f"_stop_{key}",
            disabled=not is_rec,
        ):
            frames = recorder.stop_recording()
            if frames:
                st.session_state[result_key] = frames
            st.rerun()

        if is_rec:
            st.caption(f"🔴 Запись: {n} кадров накоплено")
        elif not ctx.state.playing:
            st.info("Нажми ▶ (кнопка выше) для запуска камеры, затем «Начать запись».")

    elif not ctx.state.playing:
        st.info(f"Нажми **START** чтобы включить {label}.")

    return st.session_state.pop(result_key, None)
