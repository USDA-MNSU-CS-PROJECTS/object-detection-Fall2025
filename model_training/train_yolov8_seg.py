# train_yolov8_seg.py
import os
from ultralytics import YOLO

# 1. Path to dataset YAML file
DATA_CONFIG = "data.yaml"  # Update this path if your dataset is elsewhere

# 2. Model name - can use a pretrained YOLOv8 segmentation model
MODEL_NAME = "yolov8n-seg.pt"  # small, fast model for testing

# 3. Output directory (optional)
OUTPUT_DIR = "runs/train/exp_test"

def main():
    print("🚀 Starting YOLOv8 segmentation training on HPC...")
    print(f"Using dataset: {DATA_CONFIG}")

    # 4. Load the YOLO model
    model = YOLO(MODEL_NAME)

    # 5. Train
    model.train(
        data=DATA_CONFIG,
        epochs=10,            # For test run
        imgsz=640,            # Image size
        batch=4,              # Adjust based on GPU memory
        device=0,             # GPU id
        project="runs/train",
        name="exp_test",
        exist_ok=True
    )

    print(f"✅ Training complete. Check output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()