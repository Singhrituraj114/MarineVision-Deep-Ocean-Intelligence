from __future__ import annotations

import base64
from io import BytesIO

import streamlit.components.v1 as components
from PIL import Image


def _to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_image_compare(original: Image.Image, annotated: Image.Image, height: int | None = None) -> None:
    """Renders a draggable before/after reveal slider comparing the original and annotated frame.

    If `height` is not provided, the iframe height will be derived from the taller image
    (capped to a sensible maximum) to ensure the full image is visible.
    """
    before_uri = _to_data_uri(original)
    after_uri = _to_data_uri(annotated)

    # Compute aspect ratio from the provided images and create a responsive
    # container that preserves that ratio. This avoids iframe-based zooming
    # or cropping and ensures both images overlay exactly.
    try:
        img_w = original.width if original.width else 1
        img_h = original.height if original.height else 1
    except Exception:
        img_w, img_h = 1, 1

    aspect_pct = (img_h / img_w) * 100

    components.html(
        f"""
        <script type="module" src="https://unpkg.com/img-comparison-slider@8/dist/index.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@8/dist/styles.css" />
        <style>
          html, body {{ margin: 0; background: transparent; }}
          /* Responsive aspect-ratio wrapper: uses padding-top to preserve aspect */
          .mv-compare-wrapper {{
            position: relative;
            width: 100%;
            max-width: 100%;
            padding-top: {aspect_pct}%;
            border-radius: 1.25rem;
            overflow: hidden;
            border: 1px solid rgba(125, 211, 252, 0.25);
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
            background: transparent;
          }}
          /* absolutely fill the aspect box */
          .mv-compare-wrapper > img-comparison-slider {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            display: block;
          }}
          /* Make each image fill the slider while preserving content (no crop) */
          img-comparison-slider img {{
            position: absolute;
            inset: 0;
            width: 100% !important;
            height: 100% !important;
            display: block;
            object-fit: contain !important;
            object-position: center center !important;
          }}
          .mv-label {{
            position: absolute;
            top: 10px;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            background: rgba(2, 17, 27, 0.72);
            color: #e6f7ff;
            font: 700 12px/1 -apple-system, sans-serif;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            z-index: 2;
            pointer-events: none;
          }}
          .mv-label--before {{ left: 10px; }}
          .mv-label--after {{ right: 10px; }}
        </style>
        <div class="mv-compare-wrapper">
          <span class="mv-label mv-label--before">Before</span>
          <span class="mv-label mv-label--after">After</span>
          <img-comparison-slider>
            <img slot="first" src="{before_uri}" />
            <img slot="second" src="{after_uri}" />
          </img-comparison-slider>
        </div>
        """,
        height= int(max(300, min(1400, (aspect_pct / 100.0) * 900))) ,
    )
