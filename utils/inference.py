from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class DetectionResult:
    original_image: Image.Image
    annotated_image: Image.Image
    detections: pd.DataFrame
    inference_time: float

    @property
    def total_detections(self) -> int:
        return int(len(self.detections))

    @property
    def unique_classes(self) -> int:
        return int(self.detections["class_name"].nunique()) if not self.detections.empty else 0

    @property
    def average_confidence(self) -> float:
        return float(self.detections["confidence"].mean()) if not self.detections.empty else 0.0

    @property
    def top_class(self) -> str:
        if self.detections.empty:
            return "No detections"
        return str(self.detections["class_name"].value_counts().idxmax())


def _resolve_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return f"class_{class_id}"


def _image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def run_detection(
    model: Any,
    image: Image.Image,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> DetectionResult:
    input_image = image.convert("RGB")
    image_array = _image_to_array(input_image)

    started = time.perf_counter()
    results = model.predict(source=image_array, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    inference_time = time.perf_counter() - started

    result = results[0]
    boxes = result.boxes
    rows: list[dict[str, Any]] = []

    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        class_names = getattr(model, "names", {})

        for idx, class_id in enumerate(classes):
            x1, y1, x2, y2 = xyxy[idx]
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": _resolve_class_name(class_names, int(class_id)),
                    "confidence": float(confs[idx]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "area": float((x2 - x1) * (y2 - y1)),
                }
            )

    detections = pd.DataFrame(
        rows,
        columns=[
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "area",
        ],
    )

    annotated = result.plot()
    if annotated is None:
        annotated_image = input_image
    else:
        annotated_array = annotated[:, :, ::-1] if annotated.ndim == 3 else annotated
        annotated_image = Image.fromarray(annotated_array.astype(np.uint8))

    return DetectionResult(
        original_image=input_image,
        annotated_image=annotated_image,
        detections=detections,
        inference_time=inference_time,
    )
