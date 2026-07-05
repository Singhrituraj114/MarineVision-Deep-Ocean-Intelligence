from __future__ import annotations

from pathlib import Path

import streamlit as st

from .constants import MODEL_PATH


@st.cache_resource(show_spinner=False)
def load_detection_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the trained model file at {path}. "
            "Place MarineVision_YOLOv11l_best.pt in the models/ directory."
        )

    from ultralytics import YOLO

    return YOLO(str(path))
