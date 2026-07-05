from __future__ import annotations

import streamlit as st

from components.metrics import render_metrics_dashboard
from utils.constants import DATASET_DETAILS, FINAL_METRICS, MODEL_PATH, TRAINING_DETAILS
from utils.ui import configure_page, load_css, render_sidebar


def _render_info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card metric-card--compact">
          <div class="metric-card__label">{title}</div>
          <div class="metric-card__subtitle">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page("MarineVision | Model Information")
    load_css()
    render_sidebar()

    st.markdown("<h1 class='hero-shell__title'>Model Information</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-shell__subtitle'>Architecture, training pipeline, dataset profile, and production metrics for the deployed checkpoint.</p>",
        unsafe_allow_html=True,
    )

    render_metrics_dashboard(FINAL_METRICS)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<h2 class='section-title'>YOLOv11l Architecture Overview</h2>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="hero-highlight">
              Ultralytics YOLOv11l was fine-tuned with transfer learning to replace the COCO head
              with a custom 34-class underwater detection head. The final deployment checkpoint is
              <strong>{MODEL_PATH.name}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_info_card("Inference profile", "GPU-accelerated, near real-time image analysis at 640 x 640 input resolution.")
        _render_info_card("Detection task", "Multi-class object detection for debris, organisms, plants, and ROV objects.")
    with right:
        st.markdown("<h2 class='section-title'>Training Details</h2>", unsafe_allow_html=True)
        for key, value in TRAINING_DETAILS.items():
            _render_info_card(key, value)

    st.markdown("<h2 class='section-title'>Dataset Details</h2>", unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for column, (key, value) in zip(cols, list(DATASET_DETAILS.items())[:3], strict=False):
        with column:
            _render_info_card(key, value)
    cols = st.columns(3, gap="medium")
    for column, (key, value) in zip(cols, list(DATASET_DETAILS.items())[3:], strict=False):
        with column:
            _render_info_card(key, value)

    st.markdown("<h2 class='section-title'>Methodology</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-highlight">
          The workflow combines transfer learning, heavy underwater-specific augmentation,
          and post-training validation. The final model was selected from the best checkpoint
          using validation mAP50 and deployed as a production-ready inference artifact.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
