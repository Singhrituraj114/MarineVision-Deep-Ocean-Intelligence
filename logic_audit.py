"""One-shot logic audit for MarineVision. Run: python logic_audit.py"""
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append((name, detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ---------- 1. is_video_file ----------
from app import is_video_file, extract_video_frames

check("is_video_file: .mp4", is_video_file("clip.mp4"))
check("is_video_file: .MOV uppercase", is_video_file("CLIP.MOV"))
check("is_video_file: .jpg rejected", not is_video_file("photo.jpg"))
check("is_video_file: no extension rejected", not is_video_file("noext"))

# ---------- 2. extract_video_frames on a real generated MP4 ----------
import cv2

tmp = os.path.join(tempfile.gettempdir(), "mv_test.mp4")
w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for i in range(50):
    w.write(np.full((240, 320, 3), i * 5 % 255, dtype=np.uint8))
w.release()
with open(tmp, "rb") as f:
    frames = extract_video_frames(io.BytesIO(f.read()), max_frames=5)
check("extract_video_frames: 5 frames from 50-frame video", len(frames) == 5, f"got {len(frames)}")
check("extract_video_frames: frames are PIL RGB", all(isinstance(fr, Image.Image) and fr.mode == "RGB" for fr in frames))

w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for i in range(3):
    w.write(np.zeros((240, 320, 3), dtype=np.uint8))
w.release()
with open(tmp, "rb") as f:
    short = extract_video_frames(io.BytesIO(f.read()), max_frames=5)
check("extract_video_frames: short video returns its 3 frames", len(short) == 3, f"got {len(short)}")
os.unlink(tmp)

# ---------- 3. _resolve_class_name ----------
from utils.inference import DetectionResult, _resolve_class_name, image_to_bytes, run_detection

check("resolve: dict hit", _resolve_class_name({7: "trash_bag"}, 7) == "trash_bag")
check("resolve: dict miss fallback", _resolve_class_name({7: "x"}, 9) == "class_9")
check("resolve: list hit", _resolve_class_name(["a", "b"], 1) == "b")
check("resolve: list out-of-range fallback", _resolve_class_name(["a"], 5) == "class_5")

# ---------- 4. DetectionResult properties ----------
img = Image.new("RGB", (100, 80))
COLS = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height", "area"]
empty_df = pd.DataFrame(columns=COLS)
r_empty = DetectionResult(img, img, empty_df, 0.1)
check("empty result: totals zero", r_empty.total_detections == 0 and r_empty.unique_classes == 0)
check("empty result: avg conf 0.0", r_empty.average_confidence == 0.0)
check("empty result: top_class placeholder", r_empty.top_class == "No detections")

df = pd.DataFrame(
    [
        {"class_name": "fish", "confidence": 0.9},
        {"class_name": "fish", "confidence": 0.8},
        {"class_name": "bag", "confidence": 0.7},
    ]
)
r = DetectionResult(img, img, df, 0.1)
check("result: total=3 unique=2", r.total_detections == 3 and r.unique_classes == 2)
check("result: avg conf 0.8", abs(r.average_confidence - 0.8) < 1e-9)
check("result: top_class majority", r.top_class == "fish")

# ---------- 5. PNG serialization ----------
check("image_to_bytes: valid PNG", image_to_bytes(img)[:8] == b"\x89PNG\r\n\x1a\n")
from components.analytics import _to_png_bytes

check("_to_png_bytes: valid PNG", _to_png_bytes(img)[:8] == b"\x89PNG\r\n\x1a\n")

# ---------- 6. Chart builders ----------
from utils.visuals import build_class_frequency_figure, build_confidence_figure, overlay_detections

check("confidence fig: empty df -> placeholder", len(build_confidence_figure(empty_df).layout.annotations) == 1)
det_df = pd.DataFrame(
    [
        {"class_name": "fish", "confidence": 0.9, "x1": 5, "y1": 5, "x2": 50, "y2": 40},
        {"class_name": "bag", "confidence": 0.6, "x1": 10, "y1": 10, "x2": 30, "y2": 30},
    ]
)
check("confidence fig: has data", len(build_confidence_figure(det_df).data) >= 1)
check("class freq fig: empty df -> placeholder", len(build_class_frequency_figure(empty_df).layout.annotations) == 1)
check("class freq fig: has data", len(build_class_frequency_figure(det_df).data) >= 1)

# ---------- 7. overlay_detections ----------
base = Image.new("RGB", (100, 80), (0, 40, 60))
out = overlay_detections(base, det_df, None)
check("overlay: primary only, same size RGB", out.size == base.size and out.mode == "RGB")
check("overlay: drew boxes (image changed)", np.asarray(out).sum() != np.asarray(base).sum())
check("overlay: primary+secondary works", overlay_detections(base, det_df, det_df.assign(confidence=0.1)).size == base.size)
check("overlay: empty df -> unchanged image", np.array_equal(np.asarray(overlay_detections(base, empty_df, None)), np.asarray(base)))

print("\n--- loading model for end-to-end checks ---")
from utils.constants import MODEL_PATH, SUPPORTED_CLASSES
from utils.model_loader import load_detection_model

m1 = load_detection_model(MODEL_PATH)
m2 = load_detection_model(MODEL_PATH)
check("model cache: same object twice", m1 is m2)
check("model: exactly 34 classes", len(m1.names) == 34, f"got {len(m1.names)}")
check("model: class names match constants", sorted(m1.names.values()) == sorted(SUPPORTED_CLASSES))

# ---------- 8. Real detection, low conf so boxes appear ----------
rng = np.random.default_rng(7)
scene = Image.fromarray((rng.random((480, 640, 3)) * 255).astype("uint8"))
res = run_detection(m1, scene, conf=0.001, iou=0.45)
check("run_detection: all 10 columns", list(res.detections.columns) == COLS)
if not res.detections.empty:
    d = res.detections
    check("run_detection: boxes valid (x2>x1, y2>y1)", bool(((d.x2 > d.x1) & (d.y2 > d.y1)).all()))
    check("run_detection: confidences in (0,1]", bool(((d.confidence > 0) & (d.confidence <= 1)).all()))
    check("run_detection: area == w*h", bool(np.allclose(d.area, d.width * d.height)))
    check("run_detection: class names resolved", bool(d.class_name.isin(SUPPORTED_CLASSES).all()))
else:
    check("run_detection: low-conf produced boxes", False, "0 detections at conf=0.001")
check("run_detection: annotated same size", res.annotated_image.size == scene.size)
check("run_detection: timing positive", res.inference_time > 0)
res_hi = run_detection(m1, scene, conf=0.9, iou=0.45)
check("conf threshold filters detections", res_hi.total_detections <= res.total_detections)

# ---------- 9. Webcam processor frame-skip + stats ----------
import av

from components.webcam import MarineVisionVideoProcessor

proc = MarineVisionVideoProcessor(model=m1, conf=0.001, iou=0.45, frame_skip=3)
arr = np.asarray(scene)
first_out = None
for i in range(6):
    vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
    first_out = proc.recv(vf) if first_out is None else proc.recv(vf)
check("webcam recv: output frame same shape", first_out.to_ndarray(format="rgb24").shape == arr.shape)
stats = proc.snapshot_stats()
check("webcam frame-skip: 3 inferences over 6 frames", stats["frames_seen"] == 3, f"got {stats['frames_seen']}")
check("webcam stats: total_detections is int", isinstance(stats["total_detections"], int))
check("webcam stats: top_class present", stats["top_class"] != "")
check("webcam stats: fps positive", stats["fps"] > 0)

print(f"\n===== {len(passed)} passed, {len(failed)} failed =====")
for name, detail in failed:
    print("FAILED:", name, detail)
sys.exit(1 if failed else 0)
