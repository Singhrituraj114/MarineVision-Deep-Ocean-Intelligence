# MarineVision Deep Ocean Intelligence

AI-powered underwater object detection and analytics built with Streamlit and a custom YOLOv11l checkpoint.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## Overview

MarineVision Deep Ocean Intelligence is a professional underwater computer vision platform for detecting marine debris, marine life, vegetation, and ROV/equipment objects in images, videos, and live webcam streams.

The application combines:

- a custom-trained YOLOv11l checkpoint
- a polished Streamlit interface
- interactive analytics dashboards
- detailed dataset and class documentation
- real-time image, video, and webcam inference

## Project Goals

| Goal | Description |
|---|---|
| Marine debris detection | Identify underwater waste and human-made objects |
| Marine environment monitoring | Assist conservation and inspection workflows |
| Real-time usability | Support fast batch and live inference in Streamlit |
| Decision support | Present annotated outputs and analytics for review |

## Key Highlights

| Highlight | Value |
|---|---|
| Dataset size | 17,145 images |
| Annotated objects | 46,515 objects |
| Number of classes | 34 classes |
| Model architecture | YOLOv11l |
| Final mAP50 | 85.14% |
| Final mAP50-95 | 64.49% |
| Precision | 80.98% |
| Recall | 81.12% |

## How It Works

1. The user uploads an image, video, or starts the webcam.
2. The app loads the checkpoint from `models/MarineVision_YOLOv11l_best.pt`.
3. Inference runs through `utils/inference.py`.
4. The model returns detections with bounding boxes, confidence scores, and class labels.
5. `components/analytics.py` renders the annotated image, charts, tables, debug overlays, and comparison views.
6. Results can be downloaded or inspected in the interactive dashboard.

## System Architecture

| Layer | Responsibility |
|---|---|
| UI layer | Streamlit pages, detection controls, analytics panels, sidebar navigation |
| Inference layer | YOLOv11l model loading, prediction, annotation generation |
| Visualization layer | Plotly charts, result tables, overlays, and comparison views |
| Data layer | Dataset constants, class metadata, configuration, and model path |

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Object Detection | Ultralytics YOLOv11l |
| Deep Learning | PyTorch |
| Charts | Plotly |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |
| Live Webcam | streamlit-webrtc |

## Dataset Details

The app is based on the **Underwater Trash Detection Dataset**.

### Dataset Summary

| Field | Value |
|---|---|
| Total images | 17,145 |
| Total objects | 46,515 |
| Total classes | 34 |
| Train images | 14,523 |
| Validation images | 1,877 |
| Test images | 745 |
| Empty label files | 1,049 |
| Mean object area | 0.005658 |
| Class imbalance | Yes (51-6569 instances) |

### Why This Dataset Matters

Underwater scenes are difficult for standard detectors because of:

- poor visibility
- color distortion from water absorption
- scattering and blur
- partial occlusion by sediment, plants, or rocks
- heavy class imbalance
- small objects with low contrast

This makes an underwater-specific dataset essential for reliable real-world monitoring.

## Dataset Split

| Split | Images | Percentage |
|---|---:|---:|
| Training | 14,523 | 84.8% |
| Validation | 1,877 | 10.9% |
| Testing | 745 | 4.3% |

## Class Taxonomy

The project supports 34 classes total, including four legacy compatibility entries.

### Class Groups

| Group | Classes | Count |
|---|---|---:|
| Marine Life | `animal_crab`, `animal_eel`, `animal_etc`, `animal_fish`, `animal_shells`, `animal_starfish` | 6 |
| Flora | `plant` | 1 |
| Equipment | `rov` | 1 |
| Containers | `trash_bag`, `trash_bottle`, `trash_can`, `trash_container`, `trash_cup` | 5 |
| Textiles | `trash_fabric`, `trash_fishing_gear`, `trash_net` | 3 |
| Metal | `trash_metal`, `trash_pipe`, `trash_wreckage` | 3 |
| Organic | `trash_wood`, `trash_branch` | 2 |
| Paper | `trash_paper`, `trash_snack_wrapper`, `trash_tarp` | 3 |
| String / Rope | `trash_rope`, `trash_fishing_gear` | 2 |
| Plastic | `trash_plastic`, `trash_rubber` | 2 |
| Other | `trash_etc`, `trash_unknown_instance` | 2 |
| Legacy Metadata | `class_0`, `class_1`, `class_2`, `class_3` | 4 |

### Full Class List

| Class | Category |
|---|---|
| `animal_crab` | Marine Life |
| `animal_eel` | Marine Life |
| `animal_etc` | Marine Life |
| `animal_fish` | Marine Life |
| `animal_shells` | Marine Life |
| `animal_starfish` | Marine Life |
| `plant` | Flora |
| `rov` | Equipment |
| `trash_bag` | Containers |
| `trash_bottle` | Containers |
| `trash_branch` | Organic |
| `trash_can` | Containers |
| `trash_clothing` | Textiles |
| `trash_container` | Containers |
| `trash_cup` | Containers |
| `trash_etc` | Other |
| `trash_fabric` | Textiles |
| `trash_fishing_gear` | String / Rope |
| `trash_metal` | Metal |
| `trash_net` | Textiles |
| `trash_paper` | Paper |
| `trash_pipe` | Metal |
| `trash_plastic` | Plastic |
| `trash_rope` | String / Rope |
| `trash_rubber` | Plastic |
| `trash_snack_wrapper` | Paper |
| `trash_tarp` | Paper |
| `trash_unknown_instance` | Other |
| `trash_wood` | Organic |
| `trash_wreckage` | Metal |
| `class_0` | Legacy Metadata |
| `class_1` | Legacy Metadata |
| `class_2` | Legacy Metadata |
| `class_3` | Legacy Metadata |

## Model and YOLOv11l Details

### Why YOLOv11l

YOLOv11l is used because it offers a practical balance between:

- high detection accuracy
- real-time speed
- strong feature extraction
- support for multi-class underwater detection

### How YOLO Works

YOLO (You Only Look Once) is a one-stage detector that:

1. processes the image in a single forward pass
2. extracts feature maps through a backbone
3. fuses multi-scale context through a neck
4. predicts boxes and class probabilities from a head
5. returns final detections after confidence filtering and NMS

### YOLOv11l Pipeline in This Project

| Stage | Role |
|---|---|
| Backbone | Extracts low-level and high-level features |
| Neck | Combines features across scales |
| Head | Predicts class probabilities and boxes |
| Post-processing | Filters predictions using confidence and IoU thresholds |

### Training Setup

| Parameter | Value |
|---|---|
| Framework | Ultralytics YOLO |
| Model | YOLOv11l |
| Input size | 640 x 640 |
| Epochs | 90 |
| Batch size | 32 |
| Workers | 4 |
| Optimizer | Auto |
| Hardware | Dual NVIDIA T4 GPUs |
| Checkpoint | `MarineVision_YOLOv11l_best.pt` |

### Transfer Learning Idea

The model starts from pretrained YOLO weights and is fine-tuned on underwater data. This helps because:

- the model already understands general object features
- training converges faster
- fewer underwater-specific examples are needed than training from scratch
- the detection head can be adapted to 34 custom classes

## Training and Evaluation

| Metric | Value | Interpretation |
|---|---:|---|
| mAP50 | 85.14% | Strong standard detection quality |
| mAP50-95 | 64.49% | Good performance under stricter IoU thresholds |
| Precision | 80.98% | Low false-positive rate |
| Recall | 81.12% | Good object coverage |

### Why These Metrics Matter

- **Precision** matters when false alarms are expensive.
- **Recall** matters when missed detections are dangerous or undesirable.
- **mAP50** is a common benchmark for general object detection quality.
- **mAP50-95** gives a stricter and more realistic quality view.

## Application Workflow

### Batch Upload Flow

| Step | Action |
|---|---|
| 1 | Upload one or more images or videos |
| 2 | Select confidence and IoU thresholds |
| 3 | Run inference on each image or frame |
| 4 | View annotated output |
| 5 | Inspect the detection table and charts |
| 6 | Download annotated images |

### Live Webcam Flow

| Step | Action |
|---|---|
| 1 | Open the Live Webcam tab |
| 2 | Allow camera access |
| 3 | The video stream is processed frame by frame |
| 4 | Live detections are returned |
| 5 | The stats panel updates with frame and detection counts |

## Analytics and Debug Tools

| Tool | Purpose |
|---|---|
| Detection table | Lists boxes, classes, confidence, and geometry |
| Confidence chart | Shows score distribution across detections |
| Class frequency chart | Shows which classes appear most often |
| Side-by-side preview | Compares the original and annotated image |
| Low-confidence debug mode | Reveals borderline predictions to diagnose missed detections |
| Overlay mode | Draws high- and low-confidence boxes together |

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `components/` | UI modules and result rendering |
| `pages/` | Multipage Streamlit documentation and dashboards |
| `utils/` | Model loading, inference, constants, visualization helpers |
| `assets/` | Styles and static UI assets |
| `models/` | Local model checkpoint folder |

## How the Codebase Is Organized

- `utils/constants.py` stores dataset, class, and model metadata.
- `utils/model_loader.py` loads and caches the YOLO model.
- `utils/inference.py` runs inference and formats results.
- `utils/visuals.py` builds charts and debug overlays.
- `components/analytics.py` renders the output dashboard.
- `components/webcam.py` powers live webcam inference.
- `components/detection.py` contains the upload and threshold controls.
- `app.py` connects the entire workflow.

## UI Design Language

The interface uses:

- a deep ocean background
- cyan and teal accents
- glassmorphism cards
- rounded panels and soft shadows
- subtle motion and transitions
- responsive layouts for desktop and mobile

This makes the app feel like a premium marine intelligence product rather than a plain demo.

## Installation

### Requirements

- Python 3.10+
- pip
- the local model file at `models/MarineVision_YOLOv11l_best.pt`

### Setup

```bash
git clone https://github.com/<your-username>/MarineVision-Deep-Ocean-Intelligence.git
cd MarineVision-Deep-Ocean-Intelligence
python -m venv venv
venv\Scripts\activate  # Windows
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py
```

## Important Notes

- The checkpoint file is intentionally kept local only.
- The app will not run detection if the checkpoint is missing.
- Large model files are not committed to GitHub.
- The repository focuses on code, documentation, and reproducibility.

## Summary

MarineVision Deep Ocean Intelligence is a full underwater object detection system built around a custom YOLOv11l checkpoint, a carefully described dataset, a structured class taxonomy, and a professional Streamlit analytics interface.

It is designed to be useful for:

- marine debris monitoring
- underwater inspection workflows
- environmental analysis
- robotics support
- research and demonstration
