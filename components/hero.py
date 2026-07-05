from __future__ import annotations

import streamlit as st

from utils.constants import FINAL_METRICS, PROJECT_HIGHLIGHTS, PROJECT_NAME, PROJECT_TAGLINE


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="hero-shell__glow hero-shell__glow--one"></div>
          <div class="hero-shell__glow hero-shell__glow--two"></div>
          <div class="hero-shell__badge">{PROJECT_NAME} • YOLOv11l</div>
          <h1 class="hero-shell__title">{PROJECT_TAGLINE}</h1>
          <p class="hero-shell__subtitle">
            A premium underwater intelligence platform that detects marine debris,
            organisms, plants, and ROV objects from underwater imagery in real time.
          </p>
          <div class="hero-shell__chips">
            <span>34 custom classes</span>
            <span>{FINAL_METRICS["mAP50"]} mAP50</span>
            <span>Dual NVIDIA T4 training</span>
            <span>640 x 640 inference</span>
          </div>
          <div class="hero-shell__highlights">
            {"".join(f"<div class='hero-highlight'>{item}</div>" for item in PROJECT_HIGHLIGHTS)}
          </div>
          <div class="hero-shell__wave"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        with st.container(key="hero_cta_primary"):
            st.page_link("pages/01_Model_Information.py", label="Explore model intelligence", icon="🧠")
    with col2:
        with st.container(key="hero_cta_secondary"):
            st.page_link("pages/04_Dataset_Overview.py", label="View dataset overview", icon="📊")
