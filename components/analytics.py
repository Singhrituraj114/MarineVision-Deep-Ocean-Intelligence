from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
import uuid

# compare_slider is no longer used here; we show annotated image directly
from typing import Any
from utils.inference import DetectionResult, run_detection
from utils.visuals import overlay_detections
from utils.visuals import build_class_frequency_figure, build_confidence_figure


def _to_png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_detection_results(result: DetectionResult, model: Any | None = None, key_prefix: str | None = None) -> None:
    st.markdown("<h2 class='section-title'>Advanced Analytics Panel</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Inspect detection density, confidence patterns, and class frequency.</p>",
        unsafe_allow_html=True,
    )

    metrics = st.columns(4, gap="medium")
    summary_items = [
        ("Detections", str(result.total_detections), "Total objects found"),
        ("Classes", str(result.unique_classes), "Unique classes present"),
        ("Avg confidence", f"{result.average_confidence:.2%}", "Mean score"),
        ("Inference time", f"{result.inference_time:.2f}s", "Per image"),
    ]

    for column, (label, value, subtitle) in zip(metrics, summary_items, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="metric-card metric-card--compact">
                  <div class="metric-card__label">{label}</div>
                  <div class="metric-card__value">{value}</div>
                  <div class="metric-card__subtitle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show annotated image (always) instead of a before/after slider.
    st.markdown("<div class='panel-heading'>Annotated result</div>", unsafe_allow_html=True)
    # Use the image's native width up to a sensible maximum to avoid deprecation warnings
    img_w = getattr(result.annotated_image, "width", None) or None
    display_w = min(img_w, 900) if img_w else None
    st.image(result.annotated_image, width=display_w)

    # Ensure widget keys are unique per image/frame
    if key_prefix is None:
        key_prefix = f"kp_{uuid.uuid4().hex[:8]}"

    # Optional side-by-side comparison
    side_by_side = st.checkbox("Show side-by-side original and annotated", value=False, key=f"sidebyside_{key_prefix}")
    if side_by_side:
        cols = st.columns(2, gap="large")
        with cols[0]:
            orig_w = getattr(result.original_image, "width", None) or None
            st.markdown("<div class='panel-heading'>Original</div>", unsafe_allow_html=True)
            st.image(result.original_image, width=min(orig_w, 450) if orig_w else None)
        with cols[1]:
            ann_w = getattr(result.annotated_image, "width", None) or None
            st.markdown("<div class='panel-heading'>Annotated</div>", unsafe_allow_html=True)
            st.image(result.annotated_image, width=min(ann_w, 450) if ann_w else None)

    st.download_button(
        label="Download annotated image",
        data=_to_png_bytes(result.annotated_image),
        file_name="marinevision_annotated.png",
        mime="image/png",
        use_container_width=False,
        key=f"download_annotated_{key_prefix}_{uuid.uuid4().hex[:8]}",
    )

    # If no detections were produced, provide actionable guidance to the user.
    if result.detections.empty:
        st.warning(
            "No detections were found for this image. Try lowering the confidence threshold in the 'Detection Lab' settings and re-run detection."
        )

    # Debug: allow re-running with a very low confidence to reveal borderline boxes
    debug = st.checkbox("Show low-confidence boxes (debug)", value=False, key=f"debug_{key_prefix}", help="Re-run detection with a very low confidence to visualize weak predictions.")
    if debug:
        if model is None:
            st.info("Model not available here — ensure the caller passes the loaded model to enable debug re-run.")
        else:
            with st.spinner("Running low-confidence pass (debug)…"):
                low_result = run_detection(model, result.original_image, conf=0.01, iou=0.1, imgsz=640)
            st.markdown("<div class='panel-heading'>Low-confidence overlay (debug)</div>", unsafe_allow_html=True)
            lw = getattr(low_result.annotated_image, "width", None) or None
            st.image(low_result.annotated_image, width=min(lw, 900) if lw else None)
            if low_result.detections.empty:
                st.info("No low-confidence detections were found either.")
            else:
                st.success(f"Found {low_result.total_detections} detections at low threshold (debug).")
            # Combine and show overlay: primary are the original (high-conf) boxes,
            # secondary are the low-confidence boxes we just computed.
            overlay_img = overlay_detections(result.original_image, result.detections, low_result.detections)
            ow = getattr(overlay_img, "width", None) or None
            st.markdown("<div class='panel-heading'>Overlay: high (cyan) + low (amber)</div>", unsafe_allow_html=True)
            st.image(overlay_img, width=min(ow, 900) if ow else None)

    tabs = st.tabs(["Detection table", "Confidence chart", "Class frequency"])
    with tabs[0]:
        if result.detections.empty:
            st.info("No detections were produced for this image.")
        else:
            display_df = result.detections.copy()
            display_df["confidence"] = display_df["confidence"].map(lambda value: f"{value:.2%}")
            display_df["area"] = display_df["area"].map(lambda value: f"{value:.1f}")
            st.dataframe(display_df, use_container_width=True, height=380, hide_index=True, key=f"df_{key_prefix}")
    with tabs[1]:
        st.plotly_chart(build_confidence_figure(result.detections), use_container_width=True, key=f"confchart_{key_prefix}")
    with tabs[2]:
        st.plotly_chart(build_class_frequency_figure(result.detections), use_container_width=True, key=f"classchart_{key_prefix}")
