# Model Inference

This directory contains scripts for running inference using trained YOLOv8 models.

## Files

*   **`predict.py`**:
    *   **Description**: A simple script to load a trained YOLOv8 model and run prediction on a set of images.
    *   **Usage**: Run this script directly.
    *   **Dependencies**: Requires `ultralytics`.
    *   **Note**: The model path defaults to `sample_trained_models/` + `MODEL_B_VASCULAR_BUNDLES` from [`src/app/config/inference_constants.py`](../app/config/inference_constants.py) (vascular bundles weights). Input images use the `test_images` folder at the project root; adjust in `predict.py` if needed.
