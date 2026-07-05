"""Dataset Overview page - detailed dataset statistics and visualizations."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.constants import DATASET_DETAILS, TRAINING_DETAILS
from utils.ui import configure_page, load_css, render_sidebar


def render_dataset_overview() -> None:
    configure_page("Dataset Overview | MarineVision")
    load_css()
    render_sidebar()

    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-shell__badge">Dataset Information</div>
          <h1 class="hero-shell__title">Underwater Trash Detection Dataset</h1>
          <p class="hero-shell__subtitle">
            17,145 images with 46,515 annotated objects across 34 classes,
            meticulously curated for marine debris detection in underwater environments.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-card__icon">📊</div>
              <div class="metric-card__label">Total Images</div>
              <div class="metric-card__value">17,145</div>
              <div class="metric-card__subtitle">Underwater scenes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-card__icon">📦</div>
              <div class="metric-card__label">Total Objects</div>
              <div class="metric-card__value">46,515</div>
              <div class="metric-card__subtitle">Annotated labels</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-card__icon">🏷️</div>
              <div class="metric-card__label">Classes</div>
              <div class="metric-card__value">34</div>
              <div class="metric-card__subtitle">Unique categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h2 class='section-title'>Dataset Split</h2>", unsafe_allow_html=True)

    split_data = {
        "Split": ["Training", "Validation", "Testing"],
        "Images": [14523, 1877, 745],
        "Percentage": [84.8, 10.9, 4.3],
    }
    df_split = pd.DataFrame(split_data)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_split = px.pie(
            values=df_split["Images"],
            labels=df_split["Split"],
            title="Dataset Distribution",
            color_discrete_sequence=["#00E5FF", "#1DE9B6", "#00B4D8"],
        )
        fig_split.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_split, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div class="sidebar-card">
              <div class="sidebar-card__label">Training Images</div>
              <div class="sidebar-card__value">14,523</div>
              <div class="sidebar-card__label" style="margin-top:1rem">Validation Images</div>
              <div class="sidebar-card__value">1,877</div>
              <div class="sidebar-card__label" style="margin-top:1rem">Test Images</div>
              <div class="sidebar-card__value">745</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h2 class='section-title'>Dataset Statistics</h2>", unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Training Config", "Data Quality"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Average Objects per Image</div>
                  <div class="sidebar-card__value">{46515/17145:.2f}</div>
                </div>
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Total Objects</div>
                  <div class="sidebar-card__value">46,515</div>
                </div>
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Image Resolution</div>
                  <div class="sidebar-card__value">Various Sizes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Total Classes</div>
                  <div class="sidebar-card__value">34</div>
                </div>
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Class Categories</div>
                  <div class="sidebar-card__value">8 Categories</div>
                </div>
                <div class="sidebar-card">
                  <div class="sidebar-card__label">Empty Labels</div>
                  <div class="sidebar-card__value">1,049</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.markdown("### Training Configuration")
        training_df = pd.DataFrame(
            list(TRAINING_DETAILS.items()),
            columns=["Parameter", "Value"],
        )
        st.dataframe(training_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("### Data Quality Metrics")
        st.markdown(
            """
            - **Image Balance**: Well-distributed across 34 classes
            - **Annotation Quality**: Professional-grade bounding box annotations
            - **Empty Images**: 1,049 images with no objects (noise tolerance)
            - **Object Distribution**: Balanced class representation
            - **Validation Strategy**: 70-15-15 train-val-test split
            """
        )

    st.markdown("<h2 class='section-title'>Class Distribution</h2>", unsafe_allow_html=True)

    class_dist_data = {
        "Animal Classes": 6,
        "Flora": 1,
        "Equipment": 1,
        "Container/Bags": 5,
        "Textiles": 3,
        "Metal": 4,
        "Organic": 2,
        "Paper": 3,
        "String/Rope": 2,
        "Plastic": 2,
        "Unknown": 1,
        "Generic": 4,
    }

    fig_classes = px.bar(
        x=list(class_dist_data.keys()),
        y=list(class_dist_data.values()),
        title="Classes per Category",
        color_discrete_sequence=["#22d3ee"],
    )
    fig_classes.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        xaxis_title="Category",
        yaxis_title="Number of Classes",
    )
    fig_classes.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_classes, use_container_width=True)

    st.markdown("<h2 class='section-title'>Use Cases</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    use_cases = [
        ("🌊", "Marine Monitoring", "Track ocean cleanup progress in real-time"),
        ("🤖", "Underwater Robotics", "Enable autonomous debris detection for ROVs"),
        ("♻️", "Ocean Cleanup", "Support large-scale marine cleanup operations"),
        ("🏢", "Environmental Research", "Scientific underwater environment analysis"),
        ("📡", "Surveillance Systems", "Continuous marine monitoring and alerts"),
        ("🎯", "Policy Making", "Data-driven ocean conservation initiatives"),
    ]

    for i, (emoji, title, desc) in enumerate(use_cases):
        if i % 3 == 0 and i > 0:
            col1, col2, col3 = st.columns(3)

        with [col1, col2, col3][i % 3]:
            st.markdown(
                f"""
                <div class="hero-highlight">
                  <div style="font-size: 1.5rem; margin-bottom: 0.5rem">{emoji}</div>
                  <div style="font-weight: bold; margin-bottom: 0.3rem">{title}</div>
                  <div style="font-size: 0.9rem; color: #9fc7d7">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    render_dataset_overview()
