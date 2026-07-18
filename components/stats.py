from __future__ import annotations

from typing import Sequence

import streamlit as st


def render_project_statistics(stats: Sequence[dict[str, str]]) -> None:
    st.markdown("<h2 class='section-title'>Project Statistics</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Dataset scale, object density, and architecture at a glance.</p>",
        unsafe_allow_html=True,
    )

    # HTML must stay on one line per card: blank lines / indentation inside
    # st.markdown are parsed as Markdown code blocks and render as raw text.
    html = "".join(
        f'<div class="stat-card">'
        f'<div class="stat-card__icon">{item["icon"]}</div>'
        f'<div class="stat-card__value">{item["value"]}</div>'
        f'<div class="stat-card__label">{item["label"]}</div>'
        f"</div>"
        for item in stats
    )
    st.markdown(f"<div class='stat-grid'>{html}</div>", unsafe_allow_html=True)
