from __future__ import annotations

import streamlit as st

def render_detection_controls() -> tuple[float, float]:
    st.markdown("<h2 class='section-title'>Detection Lab</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Tune detection sensitivity, then choose batch upload or live webcam below.</p>",
        unsafe_allow_html=True,
    )

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, help="Minimum confidence score for detections")

    with st.expander("⚙️ Advanced Settings", expanded=False):
        iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05, help="Intersection over Union for NMS filtering")

    if "iou" not in st.session_state:
        iou = 0.45

    return conf, iou


def render_upload_widget():
    st.markdown(
        "<p class='section-subtitle'>Upload underwater images or videos for instant detection analysis.</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload images or videos",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        help="Batch upload: multiple images and/or videos. Live detection starts automatically.",
    )

    return uploaded_files
