"""Configuration and constants for MarineVision application."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILENAME = "MarineVision_YOLOv11l_best.pt"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Project metadata
PROJECT_NAME = "MarineVision"
PROJECT_TAGLINE = "Intelligent Underwater Marine Debris Detection"
PROJECT_DESCRIPTION = "An advanced computer vision system designed to detect underwater marine debris, marine organisms, plants, and ROV objects using deep learning and real-time object detection."

# Final model metrics
FINAL_METRICS = {
    "mAP50": "85.14%",
    "mAP50-95": "64.49%",
    "Precision": "80.98%",
    "Recall": "81.12%",
}

# Project statistics
PROJECT_STATS = [
    {"icon": "🖼️", "label": "Images", "value": "17,145"},
    {"icon": "📦", "label": "Objects", "value": "46,515"},
    {"icon": "🏷️", "label": "Classes", "value": "34"},
    {"icon": "🧠", "label": "Architecture", "value": "YOLOv11l"},
]

# Training configuration details
TRAINING_DETAILS = {
    "Framework": "Ultralytics YOLO",
    "Model": "YOLOv11l",
    "Input Size": "640 x 640",
    "Epochs": "90",
    "Batch Size": "32",
    "Optimizer": "Auto",
    "Workers": "4",
    "Hardware": "Dual NVIDIA T4 GPUs",
    "Checkpoint": MODEL_FILENAME,
}

# Dataset statistics
DATASET_DETAILS = {
    "Dataset": "Underwater Trash Detection Dataset",
    "Train Images": "14,523",
    "Validation Images": "1,877",
    "Test Images": "745",
    "Total Images": "17,145",
    "Total Classes": "34",
    "Total Objects": "46,515",
    "Empty Label Files": "1,049",
    "Class Imbalance": "Yes (51-6569 instances)",
    "Mean Object Area": "0.005658",
}

# Class definitions with metadata
LEGACY_CLASSES = ["class_0", "class_1", "class_2", "class_3"]

VALID_CLASSES = [
    "animal_crab",
    "animal_eel",
    "animal_etc",
    "animal_fish",
    "animal_shells",
    "animal_starfish",
    "plant",
    "rov",
    "trash_bag",
    "trash_bottle",
    "trash_branch",
    "trash_can",
    "trash_clothing",
    "trash_container",
    "trash_cup",
    "trash_etc",
    "trash_fabric",
    "trash_fishing_gear",
    "trash_metal",
    "trash_net",
    "trash_paper",
    "trash_pipe",
    "trash_plastic",
    "trash_rope",
    "trash_rubber",
    "trash_snack_wrapper",
    "trash_tarp",
    "trash_unknown_instance",
    "trash_wood",
    "trash_wreckage",
]

SUPPORTED_CLASSES = LEGACY_CLASSES + VALID_CLASSES

# Enhanced class metadata with categories and colors
CLASS_METADATA = {
    # Legacy Roboflow Metadata Classes (Corrupted - kept for model compatibility only)
    "class_0": {"category": "⚠️ Legacy Metadata", "emoji": "⚠️", "color": "#FF6B6B"},
    "class_1": {"category": "⚠️ Legacy Metadata", "emoji": "⚠️", "color": "#FF6B6B"},
    "class_2": {"category": "⚠️ Legacy Metadata", "emoji": "⚠️", "color": "#FF6B6B"},
    "class_3": {"category": "⚠️ Legacy Metadata", "emoji": "⚠️", "color": "#FF6B6B"},
    
    # Marine Life Classes (6)
    "animal_crab": {"category": "Marine Life", "emoji": "🦀", "color": "#FF6B6B"},
    "animal_eel": {"category": "Marine Life", "emoji": "🐍", "color": "#FF6B6B"},
    "animal_etc": {"category": "Marine Life", "emoji": "🐟", "color": "#FF6B6B"},
    "animal_fish": {"category": "Marine Life", "emoji": "🐟", "color": "#FF6B6B"},
    "animal_shells": {"category": "Marine Life", "emoji": "🐚", "color": "#FF6B6B"},
    "animal_starfish": {"category": "Marine Life", "emoji": "⭐", "color": "#FF6B6B"},
    
    # Flora (1)
    "plant": {"category": "Flora", "emoji": "🌿", "color": "#4ECDC4"},
    
    # Equipment (1)
    "rov": {"category": "Equipment", "emoji": "🤖", "color": "#95E1D3"},
    
    # Containers & Bags (5)
    "trash_bag": {"category": "Containers", "emoji": "🛍️", "color": "#FF6B9D"},
    "trash_bottle": {"category": "Containers", "emoji": "🍾", "color": "#FF6B9D"},
    "trash_can": {"category": "Containers", "emoji": "🗑️", "color": "#FF6B9D"},
    "trash_container": {"category": "Containers", "emoji": "📦", "color": "#FF6B9D"},
    "trash_cup": {"category": "Containers", "emoji": "☕", "color": "#FF6B9D"},
    
    # Textiles (3)
    "trash_clothing": {"category": "Textiles", "emoji": "👕", "color": "#C44569"},
    "trash_fabric": {"category": "Textiles", "emoji": "🧵", "color": "#C44569"},
    "trash_net": {"category": "Textiles", "emoji": "🕸️", "color": "#C44569"},
    
    # Metal Objects (4)
    "trash_metal": {"category": "Metal", "emoji": "🔧", "color": "#A8DADC"},
    "trash_pipe": {"category": "Metal", "emoji": "🔩", "color": "#A8DADC"},
    "trash_wreckage": {"category": "Metal", "emoji": "⚙️", "color": "#A8DADC"},
    
    # Organic Waste (2)
    "trash_branch": {"category": "Organic", "emoji": "🪵", "color": "#8B6F47"},
    "trash_wood": {"category": "Organic", "emoji": "🪵", "color": "#8B6F47"},
    
    # Paper & Packaging (3)
    "trash_paper": {"category": "Paper", "emoji": "📄", "color": "#E8B4B8"},
    "trash_snack_wrapper": {"category": "Paper", "emoji": "🍫", "color": "#E8B4B8"},
    "trash_tarp": {"category": "Paper", "emoji": "📋", "color": "#E8B4B8"},
    
    # String & Rope (2)
    "trash_rope": {"category": "String/Rope", "emoji": "🪢", "color": "#FFB4D6"},
    "trash_fishing_gear": {"category": "String/Rope", "emoji": "🎣", "color": "#FFB4D6"},
    
    # Plastic & Rubber (2)
    "trash_plastic": {"category": "Plastic", "emoji": "♻️", "color": "#FF6B9D"},
    "trash_rubber": {"category": "Plastic", "emoji": "🛞", "color": "#FF6B9D"},
    
    # Other (2)
    "trash_etc": {"category": "Other", "emoji": "❓", "color": "#FF6B6B"},
    "trash_unknown_instance": {"category": "Unknown", "emoji": "❓", "color": "#9D4EDD"},
}

# Project highlights
PROJECT_HIGHLIGHTS = [
    "Premium glassmorphism UI",
    "Transfer-learned YOLOv11l backbone",
    "34 custom underwater classes",
    "85.14% mAP50 production checkpoint",
]

# Use cases
USE_CASES = [
    "Marine debris monitoring",
    "Ocean cleanup systems",
    "Underwater robotics",
    "ROV-assisted inspections",
    "Environmental monitoring",
    "Smart marine surveillance",
]

# Color palette
COLORS = {
    "primary": "#00E5FF",
    "secondary": "#1DE9B6",
    "accent": "#00B4D8",
    "success": "#64DD17",
    "warning": "#FFD600",
    "danger": "#FF5252",
}

# Detection settings
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 1.0
