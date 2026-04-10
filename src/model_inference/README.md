# Model Inference

This directory contains scripts for running inference using trained YOLOv8 models.

## Files

*   **`predict.py`**:
    *   **Description**: A simple script to load a trained YOLOv8 model and run prediction on a set of images.
    *   **Usage**: Run this script directly.
    *   **Dependencies**: Requires `ultralytics`.
    *   **Note**: The paths for the model (`sample_trained_models/best_multi_class_client_hpc.pt`) and input images (`test_images`) are currently hardcoded in the script and may need to be updated to match your local environment.
