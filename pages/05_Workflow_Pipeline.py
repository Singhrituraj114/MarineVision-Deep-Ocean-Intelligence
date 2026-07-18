"""Workflow Pipeline visualization page."""

from __future__ import annotations

import streamlit as st

from utils.ui import configure_page, load_css, render_footer, render_page_header, render_sidebar


def render_workflow_pipeline() -> None:
    configure_page("ML Pipeline | MarineVision")
    load_css()
    render_sidebar()

    render_page_header(
        "ML Pipeline",
        "Training & Inference Pipeline",
        "End-to-end machine learning pipeline from raw data to production inference.",
    )

    st.markdown("<h2 class='section-title'>Complete ML Workflow</h2>", unsafe_allow_html=True)

    pipeline_steps = [
        {
            "step": 1,
            "title": "Dataset Collection",
            "description": "Gather 17,145 underwater images from Underwater Trash Detection Dataset",
            "details": [
                "• Total images collected: 17,145",
                "• Total annotated objects: 46,515",
                "• Image format: YOLO format with bounding boxes",
                "• Source: Underwater Trash Detection Dataset",
            ],
        },
        {
            "step": 2,
            "title": "Data Preprocessing",
            "description": "Prepare and normalize dataset with comprehensive augmentation",
            "details": [
                "• Auto-orientation applied",
                "• Adaptive histogram equalization",
                "• Resize to 640x640 (standard YOLO input)",
                "• Multiple augmentations (flip, rotation, exposure, blur, noise)",
            ],
        },
        {
            "step": 3,
            "title": "Train/Val/Test Split",
            "description": "Stratified split into training, validation, and test sets",
            "details": [
                "• Training: 14,523 images (84.8%)",
                "• Validation: 1,877 images (10.9%)",
                "• Testing: 745 images (4.3%)",
                "• Total: 17,145 images with 46,515 objects",
            ],
        },
        {
            "step": 4,
            "title": "Model Selection",
            "description": "Choose YOLOv11l with custom 34-class detection head",
            "details": [
                "• Base model: YOLOv11l (Large variant)",
                "• Transfer learning from COCO pretrained weights",
                "• Replace 80-class COCO head with 34-class underwater head",
                "• Input resolution: 640x640",
            ],
        },
        {
            "step": 5,
            "title": "Training Configuration",
            "description": "Train on 14,523 images for 90 epochs with dual GPU",
            "details": [
                "• Batch size: 32",
                "• Epochs: 90",
                "• Hardware: Dual NVIDIA T4 GPUs (device=[0,1])",
                "• Optimizer: Auto (SGD with momentum)",
            ],
        },
        {
            "step": 6,
            "title": "Loss Computation",
            "description": "Calculate box loss, classification loss, and DFL loss",
            "details": [
                "• Box loss: Localization accuracy",
                "• Classification loss: Object class accuracy",
                "• DFL loss: Distribution focal loss",
                "• Total loss: Weighted combination of all losses",
            ],
        },
        {
            "step": 7,
            "title": "Validation & Testing",
            "description": "Evaluate on validation (1,877) and test (745) sets",
            "details": [
                "• Validation: 1,877 images (10.9% of dataset)",
                "• Test: 745 images (4.3% of dataset)",
                "• Final metrics: mAP50=85.14%, mAP50-95=64.49%",
                "• Precision=80.98%, Recall=81.12%",
            ],
        },
        {
            "step": 8,
            "title": "Model Deployment",
            "description": "Deploy best.pt checkpoint to Streamlit application",
            "details": [
                "• Model file: MarineVision_YOLOv11l_best.pt (145.67 MB)",
                "• Deployment platform: Streamlit Cloud",
                "• Real-time inference with adjustable thresholds",
                "• 34-class underwater debris detection",
            ],
        },
    ]

    # Render pipeline visualization
    cols = st.columns([0.15, 0.85])
    with cols[1]:
        st.markdown("<h3 style='color: #22d3ee; margin-top: 0'>Pipeline Stages</h3>", unsafe_allow_html=True)

    for step_data in pipeline_steps:
        cols = st.columns([0.08, 0.92])

        with cols[0]:
            st.markdown(
                f"""
                <div style="
                    width: 3rem; height: 3rem;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #00E5FF, #1DE9B6);
                    display: flex; align-items: center; justify-content: center;
                    font-weight: bold; color: #03131f; font-size: 1.2rem;
                    box-shadow: 0 8px 24px rgba(0, 229, 255, 0.3);
                ">
                    {step_data['step']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cols[1]:
            st.markdown(
                f"""
                <div class="hero-highlight">
                    <h4 style="margin: 0 0 0.5rem 0; color: #f2fbff">{step_data['title']}</h4>
                    <p style="margin: 0 0 0.8rem 0; color: #b2d7e4">{step_data['description']}</p>
                    <div style="color: #9fc7d7; font-size: 0.9rem">
                        {''.join(f'<div>{item}</div>' for item in step_data['details'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if step_data["step"] < len(pipeline_steps):
            st.markdown(
                """
                <div style="
                    width: 2px; height: 2rem;
                    background: linear-gradient(180deg, rgba(34, 211, 238, 0.5), rgba(34, 211, 238, 0));
                    margin: -1rem 0 1rem 1.5rem;
                "></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<h2 class='section-title'>Data Flow Diagram</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="hero-highlight">
              <div style="font-size: 3rem; text-align: center">📸</div>
              <div style="text-align: center; font-weight: bold; margin-top: 1rem">17,145 Images</div>
              <div style="text-align: center; color: #9fc7d7; font-size: 0.9rem; margin-top: 0.5rem">Raw underwater imagery</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="hero-highlight">
              <div style="font-size: 3rem; text-align: center">⚙️</div>
              <div style="text-align: center; font-weight: bold; margin-top: 1rem">YOLOv11l</div>
              <div style="text-align: center; color: #9fc7d7; font-size: 0.9rem; margin-top: 0.5rem">34-class detector</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="hero-highlight">
              <div style="font-size: 3rem; text-align: center">🎯</div>
              <div style="text-align: center; font-weight: bold; margin-top: 1rem">Predictions</div>
              <div style="text-align: center; color: #9fc7d7; font-size: 0.9rem; margin-top: 0.5rem">Annotated outputs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h2 class='section-title'>Key Performance Indicators</h2>", unsafe_allow_html=True)

    metric_data = [
        ("🎯", "mAP50", "85.14%", "Intersection over Union @ 0.50"),
        ("📐", "mAP50-95", "64.49%", "IoU sweep from 0.50 to 0.95"),
        ("✨", "Precision", "80.98%", "Positive prediction accuracy"),
        ("🔍", "Recall", "81.12%", "Ground truth detection rate"),
        ("⚡", "Inf. Speed", "~50ms", "Per-image inference time"),
        ("💾", "Model Size", "145.7MB", "PyTorch checkpoint size"),
    ]

    cols = st.columns(3)
    for i, (emoji, label, value, desc) in enumerate(metric_data):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-card__icon">{emoji}</div>
                  <div class="metric-card__label">{label}</div>
                  <div class="metric-card__value">{value}</div>
                  <div class="metric-card__subtitle">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<h2 class='section-title'>Training Configuration</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Architecture", "Training Parameters", "Hardware Setup"])

    with tab1:
        st.markdown(
            """
            ### YOLOv11l Architecture

            **Backbone**: Modified Darknet-inspired network
            - Input: 640x640 RGB images
            - Depth: Large (l) variant
            - Spatial Pyramid Pooling: Yes
            - Focus Layer: Yes (32x32 input processing)

            **Neck**: Feature Pyramid Network (FPN)
            - Multi-scale feature extraction
            - Cross-scale connections
            - P3, P4, P5 outputs

            **Head**: Detection Head
            - 34 class predictions
            - Objectness scoring
            - Bounding box regression
            - IoU prediction

            **Activation Functions**: SiLU (Swish)
            **Normalization**: Batch Normalization
            **Total Parameters**: ~36.2M
            """
        )

    with tab2:
        config_data = {
            "Parameter": [
                "Epochs",
                "Batch Size",
                "Image Size",
                "Optimizer",
                "Workers",
                "Hardware",
                "Training Images",
                "Validation Images",
                "Classes",
                "Transfer Learning",
            ],
            "Value": [
                "90",
                "32",
                "640x640",
                "Auto (SGD with momentum)",
                "4",
                "Dual NVIDIA T4 GPUs (device=[0,1])",
                "14,523 images",
                "1,877 images",
                "34 custom classes",
                "Pretrained YOLOv11l weights",
            ],
        }
        import pandas as pd
        df_config = pd.DataFrame(config_data)
        st.dataframe(df_config, width="stretch", hide_index=True)

    with tab3:
        st.markdown(
            """
            ### Hardware Configuration

            **Training Hardware (Kaggle Notebook)**
            - GPU: Dual NVIDIA Tesla T4 (16GB VRAM each)
            - CPU: Intel/AMD multi-core processor
            - RAM: 32GB+ VRAM
            - Storage: High-speed SSD
            - Framework: Ultralytics YOLO

            **Training Specifications**
            - Device configuration: device=[0,1] (Dual GPU)
            - Batch Size: 32 (split across 2 GPUs)
            - Workers: 4 (parallel data loading)
            - Epochs: 90 (complete passes through 14,523 images)
            - Training time: ~24-36 hours

            **Inference Hardware**
            - GPU: NVIDIA T4 or compatible (optional)
            - CPU: Modern multi-core processor
            - RAM: 8GB minimum, 16GB+ recommended
            - Storage: ~200MB for model + application

            **Deployment (Streamlit Cloud)**
            - Platform: Cloud-based execution
            - Environment: Python 3.10+
            - Container: Docker compatible
            - Scaling: Automatic load balancing
            - Availability: 24/7 global access
            """
        )

    st.markdown("<h2 class='section-title'>Performance Metrics</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        ### Validation Results

        The model achieves excellent performance across all metrics:

        - **High Precision (80.98%)**: Few false positives, reliable predictions
        - **High Recall (81.12%)**: Catches most objects in images
        - **Excellent mAP50 (85.14%)**: Strong localization quality
        - **Solid mAP50-95 (64.49%)**: Good performance at stricter IoU thresholds

        These metrics indicate the model is production-ready and suitable for:
        - Marine debris monitoring
        - Ocean cleanup operations
        - Environmental research
        - Underwater robotics
        """
    )

    render_footer()


if __name__ == "__main__":
    render_workflow_pipeline()
