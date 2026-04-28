# Model Inference

This directory contains scripts for running inference using trained YOLOv8 models.

> [!NOTE]
> This is a standalone/manual inference helper path. It is not the main production flow of the project.
> The main app runtime is the Gradio pipeline in `src/app/main.py`.

## Files

*   **`predict.py`**:
    *   **Description**: A simple script to load a trained YOLOv8 model and run prediction on a set of images.
    *   **Usage**: Run this script directly from an environment with `ultralytics` installed.
    *   **Dependencies**: Requires `ultralytics`.
    *   **Defaults**:
        *   Model path: `sample_trained_models/` + `MODEL_B_VASCULAR_BUNDLES` from [`src/app/config/inference_constants.py`](../app/config/inference_constants.py).
        *   Input source: `test_images/` in project root (`predict.py` constant `TEST_IMAGES`).
    *   **Important**: `test_images/` is not guaranteed to exist in this repo snapshot. Create that folder or edit `TEST_IMAGES` in `predict.py` to point to your local image/folder path.
    *   **Output**: Ultralytics prediction output + debug JSON in `debug_output/` (`raw_predictions_debug_from_predict.json`).
