# YOLOv8 segmentation training (Dave-bot)

This **directory** is part of the Dave-bot monorepo, not a standalone project. Repo-wide setup, the Gradio app, and portable packaging are documented in [`README.markdown`](../README.markdown) at the repository root.

These files train a YOLOv8 instance-segmentation model (historically used on a JupyterLab GPU node). Trained weights for the web UI belong under `sample_trained_models/` with names configured in `src/app/config/inference_constants.py`.

> [!IMPORTANT]
> This training folder primarily documents a **legacy single-model training path** (`Cross Section` + `Vascular Bundles`).
> The current production Gradio runtime is **dual-model** (`src/app/main.py`):
> - model A: Casparian Strip + Epidermis
> - model B: Vascular Bundles
>
> Keep this distinction explicit when using or updating this document.

## Overview

The training script targets:
- **Cross Section**: Plant tissue cross sections
- **Vascular Bundles**: Vascular bundle structures within the tissue

This is compatible with older/single-model experiments. It is not a 1:1 description of the current dual-model app runtime.

---

## Prerequisites

### On HPC (JupyterLab)
- Access to JupyterLab cluster with GPU nodes
- CUDA-compatible GPU 

### Required Software
```bash
# Python 3.8+
# PyTorch with CUDA support
# Ultralytics YOLOv8
pip install ultralytics torch torchvision
```

### Dataset Structure
Your dataset should follow this structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## Layout in this repo

```
Dave-bot/
├── README.markdown
├── model_training/                 # this folder
│   ├── model_training_readme.md    # this file
│   ├── data.yaml                   # YOLO dataset config (edit paths + class names)
│   └── train_yolov8_seg.py         # main training script
├── sample_trained_models/          # place exported .pt files for Gradio (see inference_constants)
└── src/
    └── app/                        # consumes trained weights in production
```

When you run `train_yolov8_seg.py` from `model_training/`, Ultralytics creates outputs under **`runs/train/exp_*/`** (relative to the current working directory), for example:

```
runs/train/exp_*/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
└── ...
```

---

## Quick Start

### 1. Configure Your Dataset

Edit `data.yaml`:

```yaml
# Dataset paths (absolute paths recommended)
train: /path/to/your/dataset/train/images
val: /path/to/your/dataset/val/images

# Class names
names:
  0: Cross Section
  1: Vascular Bundles
```

### 2. Configure Training Parameters

Edit `train_yolov8_seg.py` to adjust hyperparameters:

```python
results = model.train(
    data=DATA_CONFIG,
    epochs=350,
    imgsz=640,
    batch=16,
    device=2,
    # ... other parameters
)
```

### 3. Run Training

**Run in a new terminal section**
```bash
# Run training (runs on device #2)
CUDA_VISIBLE_DEVICES=2 python train_yolov8_seg.py
```

## Relationship to current app runtime

Current `src/app/main.py` expects two model files in `sample_trained_models/`:

- `casparian_epidermis.pt`
- `vascular_bundles.pt`

If you use this training flow:

- outputs under `runs/train/.../weights/best.pt` are valid YOLO artifacts,
- but you must map/export them deliberately to the app's expected filenames and class semantics,
- and ensure class names in `data.yaml` match the model role you want (legacy single-model vs current dual-model pipeline).

In short: this folder is still useful for experimentation/training, but it should be treated as a separate workflow from the production dual-model UI path unless you explicitly align classes + filenames.


## Configuration

### Dataset Configuration (`data.yaml`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `train` | Path to training images | `/home/user/dataset/train/images` |
| `val` | Path to validation images | `/home/user/dataset/val/images` |
| `names` | List of class names | `[Cross Section, Vascular Bundles]` |

**Important Notes:**
- Use absolute paths to avoid path resolution issues on HPC
- Ensure train/val split is appropriate (recommended 80/20 for dataset)
- Verify all image files have corresponding label files

### Training Script Configuration (`train_yolov8_seg.py`)

Key sections to modify:

#### 1. Dataset Path
```python
DATA_CONFIG = "data.yaml"  # Path to your data.yaml file
```

#### 2. Model Selection
```python
# Available models (from smallest to largest):
model = YOLO('yolov8n-seg.pt')  # Nano - fastest, least accurate
model = YOLO('yolov8s-seg.pt')  # Small
model = YOLO('yolov8m-seg.pt')  # Medium - good balance
model = YOLO('yolov8l-seg.pt')  # Large - more accurate
model = YOLO('yolov8x-seg.pt')  # Extra large - most accurate, slowest
```

**Recommendation for 250 images:** Start with `yolov8s-seg.pt` or `yolov8m-seg.pt`

#### 3. GPU Device
```python
device=2,  # GPU number (0, 1, 2, etc.)
# Or for multiple GPUs:
device=[0,1,2],  # Use GPUs 0, 1, and 2
```

---

## Training Parameters

### Core Parameters

| Parameter | Description | Recommended Value | Notes |
|-----------|-------------|-------------------|-------|
| `epochs` | Number of training iterations | 300-400 | More epochs = better convergence |
| `imgsz` | Input image size | 640 | Higher = more detail, slower training |
| `batch` | Batch size | 16 | Adjust based on GPU memory |
| `device` | GPU device number | 2 | Check available GPUs with `nvidia-smi` |
| `patience` | Early stopping patience | 100-200 | Stop if no improvement for N epochs |

### Augmentation Parameters (for small datasets)

```python
# Color augmentation
hsv_h=0.005,       # Hue augmentation (0-1)
hsv_s=0.3,         # Saturation augmentation (0-1)
hsv_v=0.2,         # Value/brightness augmentation (0-1)

# Geometric augmentation
degrees=3,         # Rotation range (degrees)
translate=0.03,    # Translation fraction (0-1)
scale=0.2,         # Scale variation (0-1)
fliplr=0.5,        # Horizontal flip probability

# Advanced augmentation
mosaic=0.5,        # Mosaic augmentation probability
mixup=0.0,         # MixUp augmentation (disable for small datasets)
copy_paste=0.0,    # Copy-paste augmentation (disable for small datasets)
```

**Augmentation Tips:**
- **More augmentation** = better generalization, lower confidence
- **Less augmentation** = higher confidence, risk of overfitting
- For small datasets (< 500 images), use moderate augmentation
- For confidence optimization, reduce augmentation in later training runs

### Loss Weights (for confidence tuning)

```python
cls=3.0,           # Classification loss weight (default: 0.5)
box=7.5,           # Box regression loss weight (default: 7.5)
dfl=1.5,           # Distribution focal loss weight (default: 1.5)
```

**To increase confidence:**
- Increase `cls` from 0.5 → 1.0 → 2.0 → 3.0 → 4.0
- Higher values make the model more confident in predictions
- Start conservative and increase gradually

### Regularization

```python
dropout=0.0,       # Dropout rate (0-1, 0 = no dropout)
label_smoothing=0.0,  # Label smoothing (0 = off, helps prevent overconfidence)
```

**For small datasets:**
- Set `dropout=0.0` (no dropout) if not overfitting
- Keep `label_smoothing=0.0` for maximum confidence

### Learning Rate

```python
lr0=0.005,         # Initial learning rate
lrf=0.005,         # Final learning rate (fraction of lr0)
warmup_epochs=15,  # Number of warmup epochs
cos_lr=True,       # Use cosine learning rate schedule
```

### Advanced Parameters

```python
close_mosaic=20,   # Disable mosaic after N epochs (for fine-tuning)
workers=8,         # Number of data loading workers
project="runs/train",  # Save directory
name="exp_test",   # Experiment name
exist_ok=True,     # Overwrite existing experiment
```


## Monitoring Training

### Real-time Monitoring

Training progress is displayed in the console:
```
Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
  1/350     3.45G      1.234      2.156      0.543      1.234        128        640
  2/350     3.45G      1.123      2.045      0.512      1.223        128        640
...
```

### Training Outputs

After training starts, outputs are saved to:
```
runs/train/exp_name/
├── weights/
│   ├── best.pt          # Best model by validation mAP
│   └── last.pt          # Most recent checkpoint
├── results.png          # Training curves (loss, mAP, etc.)
├── results.csv          # Metrics in CSV format
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1-confidence curve
├── P_curve.png          # Precision-confidence curve
├── R_curve.png          # Recall-confidence curve
└── PR_curve.png         # Precision-Recall curve
```

### Key Metrics to Watch

**During Training:**
- **box_loss**: Should decrease steadily
- **seg_loss**: Should decrease steadily
- **cls_loss**: Should decrease steadily
- If losses plateau, training has converged

**After Training:**
- **mAP@50**: Should be > 0.90 for good performance
- **mAP@50-95**: Should be > 0.60 for good segmentation
- **F1 Score**: Should be > 0.90 for production use
- **Confusion Matrix**: Check for class imbalances

---

## Results and Evaluation

### Viewing Results

After training completes, review:

1. **results.png**: Overall training progression
2. **confusion_matrix.png**: Classification accuracy per class
3. **F1_curve.png**: Optimal confidence threshold
4. **P_curve.png & R_curve.png**: Precision and recall vs confidence

### Running Validation

```python
from ultralytics import YOLO

# Load the trained segmentation model
model = YOLO('best (17).pt')

# Run prediction on your local images
model.predict(
    source='test_images',   # folder of images or single image path
    save=True,
    imgsz=640,
    conf=0.8
)
```



### Performance Benchmarks

Based on 250 training images:

| Metric | Target | Current Best |
|--------|--------|--------------|
| F1 Score | > 0.95 | **0.99** ✅ |
| mAP@50 | > 0.90 | **1.00** ✅ |
| mAP@50-95 | > 0.60 | **0.65-0.70** ✅ |
| Avg Confidence | > 85% | **80-85%** (improving) |

---

