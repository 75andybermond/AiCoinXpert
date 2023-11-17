"""Main module to train a YOLOv8 model on a custom dataset
Specification: Path are use locally and not in the dev container for GPU usage
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(
    data="/workspaces/AICoinXpert/algo/yolo/config.yaml", epochs=50
)  # train the model
