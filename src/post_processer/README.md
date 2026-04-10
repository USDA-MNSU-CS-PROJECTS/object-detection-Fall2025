# Post-Processor

This directory contains code for post-processing the detections from the model. The main goal is to filter and select the correct objects (e.g., vascular bundles) based on the model's output.

## Files

*   **`post_processor.py`** (formerly `post_processing_test.py`):
    *   **Description**: The current, active version of the post-processing logic.
    *   **Key Logic**:
        *   Calculates areas of detected objects.
        *   Filters vascular bundles based on their location relative to the main cross-section.
        *   Generates CSV records of the analysis.
    *   **Configuration**:
        *   `PIXEL_TO_MICRON`: A constant used to convert pixel measurements to microns. **Important**: This value is currently hardcoded based on metadata from 10x stitch images (`0.9785316641067333`). If you are using images with a different magnification or resolution, this value **must be updated**.

*   **`post_process_legacy.py`** (formerly `post_process_test.py`):
    *   **Description**: An older version of the post-processing logic. Kept for reference.
    *   **Key Logic**: Contains functions for mask-to-polygon conversion and basic filtering based on centroids.

*   **`test.py`**:
    *   **Description**: An end-to-end script that runs model inference on a directory of images, performs post-processing (filtering vascular bundles within the main cross-section), and generates visualizations and a CSV report.
    *   **Usage**: Run this script to process a batch of images in `test_images`.
    *   **Outputs**:
        *   Visualizations in `post_processed/`
        *   Results CSV in `post_processed/results.csv`
    *   **Note**: This script contains its own copy of the post-processing logic and configuration (including `PIXEL_TO_MICRON`), similar to `post_processor.py`.
