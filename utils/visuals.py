from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#dbeafe"),
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_confidence_figure(detections: pd.DataFrame) -> go.Figure:
    if detections.empty:
        return _empty_figure("Confidence distribution", "No detections available yet.")

    fig = px.histogram(
        detections,
        x="confidence",
        nbins=20,
        title="Confidence distribution",
        color_discrete_sequence=["#22d3ee"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(2,17,27,0.6)",
        plot_bgcolor="rgba(3,22,32,0.45)",
        bargap=0.08,
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Confidence score",
        yaxis_title="Detections",
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="#073544"), color="#22d3ee"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, tickfont=dict(color="#cfefff"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, tickfont=dict(color="#cfefff"))
    fig.update_layout(font=dict(color="#dbeafe"), hovermode="closest")
    # Ensure hover tooltips always show clear text and background
    for trace in fig.data:
        # show bin range/center and count
        trace.update(
            hovertemplate="Confidence: %{x}<br>Count: %{y}<extra></extra>",
            hoverlabel=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0.75)")
        )
    return fig


def build_class_frequency_figure(detections: pd.DataFrame) -> go.Figure:
    if detections.empty:
        return _empty_figure("Class frequency", "No detections available yet.")

    class_counts = detections["class_name"].value_counts().reset_index()
    class_counts.columns = ["class_name", "count"]

    fig = px.bar(
        class_counts,
        x="class_name",
        y="count",
        title="Class frequency",
        color="count",
        color_continuous_scale=["#0f766e", "#06b6d4", "#22d3ee"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(2,17,27,0.6)",
        plot_bgcolor="rgba(3,22,32,0.45)",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Class",
        yaxis_title="Count",
        coloraxis_showscale=False,
    )
    fig.update_xaxes(tickangle=-35)
    # bar outlines and axes grid for visibility on dark background
    fig.update_traces(marker=dict(line=dict(width=1, color="#073544"), color="#22d3ee"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#cfefff"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#cfefff"))
    fig.update_layout(font=dict(color="#dbeafe"), hovermode="closest")
    # Make hovertooltips explicit and readable
    for trace in fig.data:
        trace.update(
            hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
            hoverlabel=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0.75)")
        )
    return fig


def overlay_detections(
    image, primary: pd.DataFrame, secondary: pd.DataFrame | None = None
) -> "Image.Image":
    """Return a PIL image with primary detections drawn prominently and
    optional secondary detections drawn as faint boxes for debugging.

    - `primary`: DataFrame of high-confidence detections (drawn in cyan)
    - `secondary`: DataFrame of low-confidence detections (drawn in amber)
    """
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to get a reasonable font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    def _draw_boxes(df: pd.DataFrame, color: tuple[int, int, int], width: int, alpha: int = 200):
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            # draw rectangle (outline)
            draw.rectangle([x1, y1, x2, y2], outline=color + (alpha,), width=width)
            label = f"{row.get('class_name', '')} {row.get('confidence', 0):.2f}"
            # Determine text size in a robust way across Pillow versions
            try:
                text_w, text_h = font.getsize(label)
            except Exception:
                try:
                    bbox = draw.textbbox((x1 + 3, y1 - 4), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    # Fallback approximate size
                    text_w, text_h = (len(label) * 6, 12)
            # label background
            draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color + (alpha,))
            draw.text((x1 + 3, y1 - text_h - 4), label, fill=(0, 0, 0, 255), font=font)

    # Secondary (low-confidence) in amber, thin
    if secondary is not None and not secondary.empty:
        _draw_boxes(secondary, (250, 204, 21), width=2, alpha=140)

    # Primary (high-confidence) in cyan/teal, prominent
    if primary is not None and not primary.empty:
        _draw_boxes(primary, (34, 211, 238), width=3, alpha=220)

    combined = Image.alpha_composite(base, overlay).convert("RGB")
    return combined
