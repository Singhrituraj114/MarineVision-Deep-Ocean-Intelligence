from __future__ import annotations

import random

import streamlit as st

_BUBBLE_COUNT = 16
_SEED = 42  # fixed so bubbles don't jump position on every Streamlit rerun


def render_background_fx() -> None:
    """Renders a fixed, decorative field of slowly rising bubbles behind all page content."""
    rng = random.Random(_SEED)
    bubbles = []
    for _ in range(_BUBBLE_COUNT):
        left = rng.uniform(2, 98)
        size = rng.uniform(6, 24)
        duration = rng.uniform(11, 24)
        delay = rng.uniform(-22, 0)
        opacity = rng.uniform(0.10, 0.32)
        bubbles.append(
            "<span class='mv-bubble' style='"
            f"left:{left:.2f}%; width:{size:.1f}px; height:{size:.1f}px; "
            f"animation-duration:{duration:.1f}s; animation-delay:{delay:.1f}s; opacity:{opacity:.2f};'></span>"
        )

    st.markdown(f"<div class='mv-bubble-field'>{''.join(bubbles)}</div>", unsafe_allow_html=True)
