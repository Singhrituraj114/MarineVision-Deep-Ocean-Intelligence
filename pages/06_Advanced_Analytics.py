"""Advanced Analytics page for batch analysis."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.ui import configure_page, load_css, render_sidebar


def render_advanced_analytics() -> None:
    configure_page("Advanced Analytics | MarineVision")
    load_css()
    render_sidebar()

    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-shell__badge">Analytics Suite</div>
          <h1 class="hero-shell__title">Advanced Analytics Dashboard</h1>
          <p class="hero-shell__subtitle">
            Deep insights into detection patterns, class distributions, and model performance metrics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h2 class='section-title'>Detection Statistics</h2>", unsafe_allow_html=True)

    # Sample data for demonstration
    sample_detections = {
        "Class": ["animal_fish", "trash_bottle", "plant", "trash_bag", "animal_crab", "trash_net"],
        "Count": [3245, 2891, 1567, 2134, 1023, 892],
        "Avg Confidence": [0.87, 0.82, 0.91, 0.78, 0.85, 0.79],
    }
    df_detections = pd.DataFrame(sample_detections)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        fig_freq = px.bar(
            df_detections,
            x="Class",
            y="Count",
            title="Detection Frequency by Class",
            color="Avg Confidence",
            color_continuous_scale=["#0f766e", "#06b6d4", "#22d3ee"],
        )
        fig_freq.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div class="sidebar-card">
              <div class="sidebar-card__label">Total Detections</div>
              <div class="sidebar-card__value">11,752</div>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-card__label">Unique Classes</div>
              <div class="sidebar-card__value">6</div>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-card__label">Avg Confidence</div>
              <div class="sidebar-card__value">0.84</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h2 class='section-title'>Class Performance Analysis</h2>", unsafe_allow_html=True)

    class_performance = {
        "Class": ["animal_fish", "trash_bottle", "plant", "trash_bag", "animal_crab", "trash_net"],
        "Precision": [0.89, 0.81, 0.93, 0.76, 0.88, 0.77],
        "Recall": [0.85, 0.79, 0.90, 0.74, 0.86, 0.75],
        "mAP50": [0.87, 0.80, 0.92, 0.75, 0.87, 0.76],
    }
    df_perf = pd.DataFrame(class_performance)

    fig_precision = px.bar(
        df_perf,
        x="Class",
        y=["Precision", "Recall", "mAP50"],
        title="Per-Class Performance Metrics",
        barmode="group",
        color_discrete_sequence=["#00E5FF", "#1DE9B6", "#00B4D8"],
    )
    fig_precision.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_precision, use_container_width=True)

    st.markdown("<h2 class='section-title'>Confidence Distribution</h2>", unsafe_allow_html=True)

    confidence_data = {
        "Confidence Range": [
            "0.50-0.60",
            "0.60-0.70",
            "0.70-0.80",
            "0.80-0.90",
            "0.90-1.00",
        ],
        "Detection Count": [234, 456, 892, 1567, 3245],
    }
    df_conf = pd.DataFrame(confidence_data)

    fig_conf = px.area(
        df_conf,
        x="Confidence Range",
        y="Detection Count",
        title="Confidence Score Distribution",
        color_discrete_sequence=["#22d3ee"],
    )
    fig_conf.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("<h2 class='section-title'>Model Performance Summary</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    metrics_summary = [
        ("🎯", "Weighted Precision", "80.98%"),
        ("🔍", "Weighted Recall", "81.12%"),
        ("📊", "Overall mAP50", "85.14%"),
        ("📈", "Overall mAP50-95", "64.49%"),
    ]

    for col, (emoji, label, value) in zip([col1, col2, col3, col4], metrics_summary):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-card__icon">{emoji}</div>
                  <div class="metric-card__label">{label}</div>
                  <div class="metric-card__value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<h2 class='section-title'>Category Analysis</h2>", unsafe_allow_html=True)

    categories = {
        "Category": [
            "Marine Life",
            "Containers",
            "Metal Objects",
            "Textiles",
            "Organic Waste",
            "Unknown",
        ],
        "Detections": [3245, 2891, 1567, 2134, 892, 123],
        "Avg Confidence": [0.87, 0.82, 0.85, 0.78, 0.81, 0.65],
    }
    df_cats = pd.DataFrame(categories)

    fig_cats = px.scatter(
        df_cats,
        x="Category",
        y="Detections",
        size="Avg Confidence",
        color="Avg Confidence",
        color_continuous_scale=["#0f766e", "#06b6d4", "#22d3ee"],
        title="Detection Category Analysis",
        size_max=50,
    )
    fig_cats.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_cats, use_container_width=True)

    st.markdown("<h2 class='section-title'>Inference Performance</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Speed Analysis", "Memory Usage", "Batch Performance"])

    with tab1:
        inference_times = {
            "Image Size": ["480p", "640p", "720p", "1080p"],
            "CPU Time (ms)": [120, 150, 180, 220],
            "GPU Time (ms)": [35, 50, 60, 85],
        }
        df_inf = pd.DataFrame(inference_times)

        fig_inf = px.line(
            df_inf,
            x="Image Size",
            y=["CPU Time (ms)", "GPU Time (ms)"],
            title="Inference Speed Comparison",
            markers=True,
            color_discrete_sequence=["#FF6B6B", "#22d3ee"],
        )
        fig_inf.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_inf, use_container_width=True)

    with tab2:
        st.markdown(
            """
            ### Memory Consumption

            **Model Loading**: ~150MB (PyTorch)
            **Per-Batch Processing (B=32)**:
            - Input images: ~45MB
            - Feature maps: ~80MB
            - Output tensors: ~15MB
            - Total: ~140MB per batch

            **Optimization Tips**:
            - Use smaller batch sizes for limited memory
            - Enable model quantization for mobile deployment
            - Consider ONNX conversion for lighter inference
            """
        )

    with tab3:
        batch_data = {
            "Batch Size": [1, 4, 8, 16, 32],
            "Total Time (ms)": [50, 180, 340, 650, 1200],
            "Time per Image (ms)": [50, 45, 42.5, 40.6, 37.5],
        }
        df_batch = pd.DataFrame(batch_data)

        fig_batch = px.line(
            df_batch,
            x="Batch Size",
            y="Time per Image (ms)",
            title="Batch Processing Efficiency",
            markers=True,
            color_discrete_sequence=["#1DE9B6"],
        )
        fig_batch.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_batch, use_container_width=True)

    st.markdown("<h2 class='section-title'>Insights & Recommendations</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### ✨ Model Strengths

            - **High recall on marine life** (87%+ on fish, crabs)
            - **Excellent plant detection** (93% precision)
            - **Robust container detection** (81%+ mAP50)
            - **Fast inference** (35-85ms per image)
            - **Low false positive rate** (80%+ precision overall)
            """
        )

    with col2:
        st.markdown(
            """
            #### 🎯 Areas for Improvement

            - **Textile recognition** (78% precision, needs tuning)
            - **Unknown object classification** (65% confidence)
            - **Small object detection** (underwater debris <20px)
            - **High-turbidity scenarios** (partial occlusion)
            - **Class imbalance** (rare categories need more data)
            """
        )


if __name__ == "__main__":
    render_advanced_analytics()
