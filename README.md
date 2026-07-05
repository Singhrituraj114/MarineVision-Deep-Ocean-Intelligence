# MarineVision Deep Ocean Intelligence

AI-powered underwater object detection and analytics built with Streamlit and a custom YOLOv11l checkpoint.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What this project does

MarineVision Deep Ocean Intelligence is a professional underwater computer vision app that helps you detect marine debris, marine life, vegetation, and ROV/equipment objects in images and videos.

It includes:

- batch image and video upload
- live webcam inference
- annotated detection results
- confidence distribution and class frequency analytics
- side-by-side comparison and low-confidence debug overlays
- a polished Streamlit UI with ocean-themed styling

## How it works

1. The user uploads an image, video, or starts the webcam.
2. The app loads the YOLOv11l checkpoint from `models/MarineVision_YOLOv11l_best.pt`.
3. Inference runs in `utils/inference.py`.
4. Detected boxes and confidence scores are returned as a dataframe plus an annotated image.
5. `components/analytics.py` renders metrics, tables, Plotly charts, comparison views, and debug overlays.
6. The same detection pipeline is reused for images, video frames, and live webcam input.

## Key features

- 34-class underwater object detection
- image, video, and webcam inference
- annotated output with downloadable results
- confidence histogram and class frequency charts
- low-confidence debug mode
- side-by-side original vs annotated preview
- responsive dark-ocean UI

## Tech stack

- Streamlit
- Ultralytics YOLOv11
- PyTorch
- Plotly
- OpenCV
- Pillow
- Pandas / NumPy

## Project structure

```text
MarineVision-Deep-Ocean-Intelligence/
├── app.py
├── README.md
├── DEPLOYMENT_GUIDE.md
├── requirements.txt
├── assets/
├── components/
├── docs/
├── models/
├── pages/
└── utils/
```

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/<your-username>/MarineVision-Deep-Ocean-Intelligence.git
cd MarineVision-Deep-Ocean-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Verify the model file

The checkpoint is intentionally not committed to GitHub because of its size.
Download or place it locally so this file exists:

```text
models/MarineVision_YOLOv11l_best.pt
```

## Run locally

```bash
streamlit run app.py
```

Open the app at `http://localhost:8501`.

## App pages

- `app.py` - main detection workspace
- `pages/01_Model_Information.py` - model and training summary
- `pages/02_About_Project.py` - project context and goals
- `pages/04_Dataset_Overview.py` - dataset distribution and insights
- `pages/05_Workflow_Pipeline.py` - pipeline and processing flow
- `pages/06_Advanced_Analytics.py` - analytics dashboard and deeper charts

## Detection workflow

### Batch upload

1. Upload one or more images or videos.
2. Set confidence and IoU thresholds.
3. The app runs inference and shows annotated output.
4. Use analytics, download annotated images, or enable debug overlays.

### Live webcam

1. Open the Live Webcam tab.
2. Allow camera access in the browser.
3. The model runs on incoming frames with live statistics.

## Configuration

Streamlit config is stored in `.streamlit/config.toml`.

Important settings:

- dark theme
- cyan primary color
- hidden usage stats
- sidebar navigation enabled

## Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Project Architecture](docs/PROJECT_ARCHITECTURE.md)
- [Full Project Deep Dive](docs/PROJECT_DEEP_DIVE.md)

## Acknowledgements

- Streamlit
- Ultralytics YOLO
- Plotly
- OpenCV