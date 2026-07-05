from __future__ import annotations

from typing import Mapping

import streamlit as st


def render_metrics_dashboard(metrics: Mapping[str, str]) -> None:
    st.markdown("<h2 class='section-title'>Performance Dashboard</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Validation results for the deployed MarineVision checkpoint.</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(4, gap="medium")
    metric_meta = [
        ("mAP50", "🎯", "Localization + classification"),
        ("mAP50-95", "📐", "Stricter IoU sweep"),
        ("Precision", "✨", "Prediction quality"),
        ("Recall", "🔎", "Coverage of ground truth"),
    ]

    for column, (label, icon, subtitle) in zip(columns, metric_meta, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-card__icon">{icon}</div>
                  <div class="metric-card__label">{label}</div>
                  <div class="metric-card__value">{metrics[label]}</div>
                  <div class="metric-card__subtitle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
