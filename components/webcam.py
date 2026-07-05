from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any

import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils.constants import MODEL_PATH
from utils.inference import run_detection
from utils.model_loader import load_detection_model


class MarineVisionVideoProcessor:
    """Runs YOLO detection on incoming webcam frames, skipping frames to stay real-time."""

    def __init__(self, model: Any, conf: float, iou: float, frame_skip: int = 3) -> None:
        self.model = model
        self.conf = conf
        self.iou = iou
        self.frame_skip = frame_skip

        self._lock = threading.Lock()
        self._frame_counter = 0
        self._last_annotated: np.ndarray | None = None
        self._stream_started = time.perf_counter()
        self._frames_seen = 0
        self._total_detections = 0
        self._class_counter: Counter[str] = Counter()
        self._last_inference_time = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = Image.fromarray(frame.to_ndarray(format="rgb24"))
        self._frame_counter += 1
        run_now = self._last_annotated is None or self._frame_counter % self.frame_skip == 0

        if run_now:
            result = run_detection(self.model, image, conf=self.conf, iou=self.iou)
            annotated_array = np.asarray(result.annotated_image)
            with self._lock:
                self._last_annotated = annotated_array
                self._frames_seen += 1
                self._total_detections += result.total_detections
                if not result.detections.empty:
                    self._class_counter.update(result.detections["class_name"].tolist())
                self._last_inference_time = result.inference_time
            output_array = annotated_array
        else:
            with self._lock:
                output_array = self._last_annotated

        return av.VideoFrame.from_ndarray(output_array, format="rgb24")

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(time.perf_counter() - self._stream_started, 1e-6)
            top_class = self._class_counter.most_common(1)[0][0] if self._class_counter else "—"
            return {
                "frames_seen": self._frames_seen,
                "total_detections": self._total_detections,
                "top_class": top_class,
                "inference_time": self._last_inference_time,
                "fps": self._frames_seen / elapsed,
            }


def _render_live_stats(stats: dict[str, Any]) -> None:
    cols = st.columns(4, gap="medium")
    cards = [
        ("Frames analyzed", f"{stats['frames_seen']}", "Processed by the model"),
        ("Live detections", f"{stats['total_detections']}", "Cumulative objects seen"),
        ("Top class", stats["top_class"], "Most frequent this session"),
        ("Inference FPS", f"{stats['fps']:.1f}", "Frames/sec analyzed"),
    ]
    for column, (label, value, subtitle) in zip(cols, cards, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="metric-card metric-card--compact">
                  <div class="metric-card__label">{label}</div>
                  <div class="metric-card__value">{value}</div>
                  <div class="metric-card__subtitle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_webcam_tab(conf: float, iou: float) -> None:
    st.markdown(
        """
        <div class="live-badge"><span class="live-badge__dot"></span>LIVE FEED</div>
        <p class="section-subtitle">
          Start your webcam to run real-time underwater-style detection, frame by frame,
          using the same MarineVision YOLOv11l checkpoint as batch upload.
        </p>
        """,
        unsafe_allow_html=True,
    )

    model = load_detection_model(MODEL_PATH)

    def _factory() -> MarineVisionVideoProcessor:
        return MarineVisionVideoProcessor(model=model, conf=conf, iou=iou)

    st.markdown("<div class='viewfinder'>", unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="marinevision-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=_factory,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    st.markdown("<div class='viewfinder__scanline'></div></div>", unsafe_allow_html=True)

    if ctx.video_processor:
        ctx.video_processor.conf = conf
        ctx.video_processor.iou = iou

    if ctx.state.playing:
        stats_placeholder = st.empty()
        while ctx.state.playing:
            if ctx.video_processor:
                with stats_placeholder.container():
                    _render_live_stats(ctx.video_processor.snapshot_stats())
            time.sleep(0.5)
    else:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-state__icon">📷</div>
              <div class="empty-state__title">Camera is idle</div>
              <div class="empty-state__subtitle">Click <b>Start</b> above and allow camera access to begin live detection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
