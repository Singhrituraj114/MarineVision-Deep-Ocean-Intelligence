from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.background_fx import render_background_fx

from .constants import ASSETS_DIR, MODEL_PATH


def configure_page(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if not css_path.exists():
        raise FileNotFoundError(f"Missing stylesheet: {css_path}")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    # Small script to add 'loaded'/'rendered' classes to images and plotly charts
    st.markdown(
        """
        <script>
        (function(){
          // Mark images as loaded when their naturalWidth is available
          function markImages(){
            document.querySelectorAll('.block-container img, .mv-compare-wrapper img, .stImage img').forEach(function(img){
              if(img.complete && img.naturalWidth>0){ img.classList.add('loaded'); }
              img.addEventListener('load', function(){ img.classList.add('loaded'); });
            });
          }
          // Mark plotly plots as rendered when they appear
          function markPlots(){
            document.querySelectorAll('.js-plotly-plot').forEach(function(plot){
              plot.classList.add('rendered');
            });
          }
          // Initial pass
          setTimeout(function(){ markImages(); markPlots(); }, 300);
          // Observe DOM mutations to catch dynamically added images/plots
          var observer = new MutationObserver(function(m){ markImages(); markPlots(); });
          observer.observe(document.body, { childList:true, subtree:true });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    render_background_fx()


def render_page_header(badge: str, title: str, subtitle: str) -> None:
    """Consistent hero-style header used across all secondary pages."""
    st.markdown(
        f"""
        <section class="hero-shell hero-shell--page">
          <div class="hero-shell__glow hero-shell__glow--one"></div>
          <div class="hero-shell__badge">{badge}</div>
          <h1 class="hero-shell__title">{title}</h1>
          <p class="hero-shell__subtitle">{subtitle}</p>
          <div class="hero-shell__wave"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <footer class="mv-footer">
          <div class="mv-footer__brand">MarineVision</div>
          <div class="mv-footer__tagline">Intelligent underwater marine debris detection, powered by YOLOv11l.</div>
          <div class="mv-footer__meta">
            <span>YOLOv11l &middot; 34 classes &middot; 85.14% mAP50</span>
            <span class="mv-footer__dot">•</span>
            <a href="https://github.com/Singhrituraj114" target="_blank" rel="noopener">GitHub</a>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "detection_result": None,
        "uploaded_image": None,
        "annotated_image_bytes": None,
        "latest_model_status": "Ready",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> None:
    model_ready = Path(MODEL_PATH).exists()
    status_label = "Model ready" if model_ready else "Model missing"
    status_class = "status-success" if model_ready else "status-warning"

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
          <div class="sidebar-brand__mark">MV</div>
          <div>
            <div class="sidebar-brand__title">MarineVision</div>
            <div class="sidebar-brand__subtitle">Underwater AI detection</div>
          </div>
        </div>
        <div class="sidebar-status {status_class}">{status_label}</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ Navigation")
    st.sidebar.page_link("app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/01_Model_Information.py", label="Model Information", icon="🧠")
    st.sidebar.page_link("pages/04_Dataset_Overview.py", label="Dataset Overview", icon="📊")
    st.sidebar.page_link("pages/05_Workflow_Pipeline.py", label="Workflow Pipeline", icon="🔄")
    st.sidebar.page_link("pages/06_Advanced_Analytics.py", label="Advanced Analytics", icon="📈")
    st.sidebar.page_link("pages/02_About_Project.py", label="About Project", icon="🌊")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Live Summary")
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-card__label">Model</div>
          <div class="sidebar-card__value">YOLOv11l</div>
        </div>
        <div class="sidebar-card">
          <div class="sidebar-card__label">Performance</div>
          <div class="sidebar-card__value">85.14% mAP50</div>
        </div>
        <div class="sidebar-card">
          <div class="sidebar-card__label">Classes</div>
          <div class="sidebar-card__value">34 Custom</div>
        </div>
        <div class="sidebar-card">
          <div class="sidebar-card__label">Training Data</div>
          <div class="sidebar-card__value">17,145 Images</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Quick Info")
    st.sidebar.info(
        "**MarineVision** is an enterprise-grade underwater object detection platform. "
        "Upload any underwater image to get real-time detection results with comprehensive analytics."
    )
