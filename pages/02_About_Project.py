from __future__ import annotations

import streamlit as st

from utils.constants import USE_CASES
from utils.ui import configure_page, load_css, render_sidebar


def main() -> None:
    configure_page("MarineVision | About Project")
    load_css()
    render_sidebar()

    st.markdown("<h1 class='hero-shell__title'>About the Project</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-shell__subtitle'>Why underwater debris detection matters and how MarineVision applies deep learning to solve it.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<h2 class='section-title'>Problem Context</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-highlight">
          Marine litter affects ecosystems, navigation safety, habitat health, and cleanup costs.
          Underwater scenes are visually complex, low-contrast, and noisy, which makes manual inspection
          slow and expensive. MarineVision automates detection so teams can focus on response and cleanup.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<h2 class='section-title'>Computer Vision Approach</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-card">
              <div class="sidebar-card__value">1. Collect underwater imagery</div>
              <div class="sidebar-card__label">Dataset curation and annotation</div>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-card__value">2. Preprocess and augment</div>
              <div class="sidebar-card__label">Resize, enhance, flip, rotate, blur, and inject noise</div>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-card__value">3. Train YOLOv11l</div>
              <div class="sidebar-card__label">Transfer learning on 34 custom classes</div>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-card__value">4. Deploy inference</div>
              <div class="sidebar-card__label">Upload an image and generate annotated detections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<h2 class='section-title'>Where it can be used</h2>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-highlight'>" + "".join(f"<div>• {item}</div>" for item in USE_CASES) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<h2 class='section-title'>Ocean Conservation Impact</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-highlight">
          By turning raw underwater frames into structured detections, MarineVision helps operators
          identify trash, track affected areas, support ROV missions, and accelerate cleanup planning.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
