from __future__ import annotations

from PIL import Image
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from components.analytics import render_detection_results
from components.classes import render_supported_classes
from components.detection import render_detection_controls, render_upload_widget
from components.hero import render_hero
from components.stats import render_project_statistics
from components.webcam import render_webcam_tab
from utils.constants import MODEL_PATH, PROJECT_STATS, SUPPORTED_CLASSES
from utils.inference import image_to_bytes, run_detection
from utils.model_loader import load_detection_model
from utils.ui import configure_page, init_session_state, load_css, render_footer, render_sidebar


def is_video_file(filename: str) -> bool:
    """Check if file is a video."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return os.path.splitext(filename)[1].lower() in video_extensions


def extract_video_frames(video_file, max_frames: int = 5) -> list[Image.Image]:
    """Extract frames from video file."""
    frames = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // max_frames)
        
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
        
        cap.release()
    finally:
        os.unlink(tmp_path)
    
    return frames


def render_how_it_works() -> None:
    st.markdown("<h2 class='section-title'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>From raw underwater frame to structured detections in three steps.</p>",
        unsafe_allow_html=True,
    )
    steps = [
        ("📤", "1 · Upload", "Drop in underwater images or video, or start your webcam for a live feed."),
        ("🧠", "2 · Detect", "The YOLOv11l checkpoint scans every frame for 34 classes of debris, marine life, plants, and ROV objects."),
        ("📊", "3 · Analyze", "Review annotated results, confidence charts, and class breakdowns — then export the annotated image."),
    ]
    cols = st.columns(3, gap="medium")
    for column, (icon, title, body) in zip(cols, steps, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-card__icon">{icon}</div>
                  <div class="metric-card__value" style="font-size:1.15rem">{title}</div>
                  <div class="metric-card__subtitle">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    configure_page("MarineVision | Intelligent Underwater Marine Debris Detection")
    load_css()
    init_session_state()
    render_sidebar()

    st.markdown("<div class='hero-shell-spacer'></div>", unsafe_allow_html=True)
    render_hero()

    render_project_statistics(PROJECT_STATS)
    render_how_it_works()

    conf, iou = render_detection_controls()

    upload_tab, live_tab = st.tabs(["📤 Batch Upload", "🔴 Live Webcam"])

    with upload_tab:
        uploaded_files = render_upload_widget()

        if uploaded_files:
            st.markdown("<h3 class='section-title'>Processing Uploads</h3>", unsafe_allow_html=True)
            model = load_detection_model(MODEL_PATH)

            for file_idx, uploaded_file in enumerate(uploaded_files, 1):
                st.markdown(f"<div class='section-subtitle'>File {file_idx}/{len(uploaded_files)}: {uploaded_file.name}</div>", unsafe_allow_html=True)

                if is_video_file(uploaded_file.name):
                    st.info(f"🎬 Extracting frames from video: {uploaded_file.name}")
                    frames = extract_video_frames(uploaded_file)

                    if not frames:
                        st.error(f"Could not extract frames from {uploaded_file.name}")
                        continue

                    for frame_idx, frame in enumerate(frames, 1):
                        with st.expander(f"Video Frame {frame_idx}/{len(frames)}", expanded=(frame_idx == 1)):
                            cols = st.columns([1.1, 0.9], gap="large")

                            with cols[0]:
                                st.markdown("<div class='panel-heading'>Original frame</div>", unsafe_allow_html=True)
                                frame_w = getattr(frame, "width", None) or None
                                st.image(frame, width=min(frame_w, 900) if frame_w else None)

                            with cols[1]:
                                with st.spinner(f"Detecting objects in frame {frame_idx}..."):
                                    result = run_detection(model, frame, conf=conf, iou=iou)
                                    render_detection_results(result, model=model, key_prefix=f"file{file_idx}_frame{frame_idx}")
                else:
                    try:
                        image = Image.open(uploaded_file).convert("RGB")

                        cols = st.columns([1.1, 0.9], gap="large")
                        with cols[0]:
                            st.markdown("<div class='panel-heading'>Uploaded image</div>", unsafe_allow_html=True)
                            img_w = getattr(image, "width", None) or None
                            st.image(image, width=min(img_w, 900) if img_w else None)

                        with cols[1]:
                            with st.spinner(f"Running detection on {uploaded_file.name}..."):
                                result = run_detection(model, image, conf=conf, iou=iou)
                                st.markdown(
                                    """
                                    <div class="metric-card metric-card--compact">
                                      <div class="metric-card__label">Model checkpoint</div>
                                      <div class="metric-card__subtitle">MarineVision_YOLOv11l_best.pt</div>
                                    </div>
                                    <div class="metric-card metric-card--compact" style="margin-top:0.6rem">
                                      <div class="metric-card__label">Inference settings</div>
                                      <div class="metric-card__subtitle">640 x 640 input · confidence {:.2f} · IoU {:.2f}</div>
                                    </div>
                                    """.format(conf, iou),
                                    unsafe_allow_html=True,
                                )

                        render_detection_results(result, model=model, key_prefix=f"file{file_idx}_img")
                        st.divider()

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="empty-state__icon">🌊</div>
                  <div class="empty-state__title">Drop underwater imagery to begin</div>
                  <div class="empty-state__subtitle">
                    Upload one or more images or videos above — detection starts automatically
                    and results appear here with full analytics.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with live_tab:
        render_webcam_tab(conf, iou)

    render_supported_classes(SUPPORTED_CLASSES)
    render_footer()


if __name__ == "__main__":
    main()
