# Application Code Overview

This document provides an overview of the core Python code that powers the Alfalfa Stem Object Detection Gradio application. The application is structured into a main Gradio interface (`main.py`) and API modules (`converter.py`, `predictor.py`, `dual_stem_pipeline.py`, `stem_metrics.py`, `yolo_label_export.py`, `filename_metadata.py`, `post_processor.py` legacy).

## `src/app/main.py`
-   **Purpose:** Serves as the main entry point for the Gradio application. It defines the user interface, handles file uploads, orchestrates the processing pipeline, and displays results.
-   **Key Functions:**
    -   `run_conversion(files)`: Handles the "Simple Image Converter" pipeline, converting ND2 files to PNG and zipping them for download.
    -   `run_full_pipeline(files, progress=gr.Progress())`: Converts inputs, runs **two** YOLO models (`sample_trained_models/casparian_epidermis.pt` and `vascular_bundles.pt`), runs noise stages (ring + casp), builds extended CSV, zips visualizations, generated labels, and merged geometry.
-   **Model configuration:** `src/app/config/inference_constants.py` (paths, class names, confidences). Noise PG/RR profiles: `src/app/config/noise_profiles_app.py`.
-   **File Handling:** Accepts `.nd2`, `.png`, `.jpg`, `.jpeg`, and `.zip` files. For `.zip` files, it extracts only `.nd2`, `.png`, `.jpg`, and `.jpeg` files, ignoring others and macOS metadata files (e.g., `._*`).
-   **JPEG note:** JPEG is supported, but JPEG compression is lossy. If exact pixel-level analysis matters, prefer PNG.

## `src/app/api/converter.py`
-   **Purpose:** Responsible for converting ND2 microscopy image files into PNG format directly without conversion to TIFF.
-   **Key Functions:**
    -   `ImageConverter.convert_nd2_to_png(input_path, output_dir)`: Converts a single ND2 file to PNG, handling channel orientation and 8-bit conversion.
    -   `ImageConverter.process_directory(input_dir, output_dir)`: Iterates through a directory, identifies ND2 files (ignoring macOS metadata files), and converts them to PNG.

## `src/app/api/predictor.py`
-   **Purpose:** Interfaces with the YOLO object detection model to perform predictions on images.
-   **Key Functions:**
    -   `ModelPredictor.__init__(model_path)`: Initializes the YOLO model with a specified trained model file.
    -   `ModelPredictor.batch_predict(image_paths, output_dir, conf_threshold, save)`: Runs predictions on a list of image paths, applying a confidence threshold and optionally saving images with prediction overlays.

## `src/app/api/dual_stem_pipeline.py`
-   **Purpose:** Orchestrates model A + model B outputs, YOLO label export, in-memory `detect_noise_mask` (ring and casp), merged noise union, calls `stem_metrics`, writes visualizations.

## `src/app/api/stem_metrics.py`
-   **Purpose:** Central place for geometric metrics (ring area, mean thickness via skeleton + distance transform, casp/epi areas with/without noise, VB counts and ratios).

## `src/app/api/yolo_label_export.py`
-   **Purpose:** Converts Ultralytics segmentation mask results into normalized polygon coordinates (YOLO-seg label style) for export alongside the dual pipeline.
-   **Key items:** `result_to_polygons_by_class(...)`, `write_yolo_seg_label_txt(...)` — used when writing label `.txt` files that align with `noise_deletion_clean` expectations.

## `src/app/api/filename_metadata.py`
-   **Purpose:** Parses experiment metadata from image basenames (fixed stem regex), e.g. incubation time, magnification, stitch/code fields for CSV columns.
-   **Key function:** `parse_image_filename_metadata(basename_with_ext)` — returns a dict with structured fields and `parse_ok` when the pattern matches.

## `src/app/api/post_processor.py`
-   **Purpose (legacy):** Single-model pipeline with Cross Section + Vascular Bundles; kept for reference — default Gradio flow uses `dual_stem_pipeline.py` instead.
-   **Status:** Legacy/manual reference module; not part of the default runtime path in `src/app/main.py`.
-   **Key Functions:**
    -   `PostProcessor.__init__(output_dir, pixel_to_micron_ratio)`: Initializes the post-processor, setting up output directories and the pixel-to-micron conversion ratio.
    -   `PostProcessor.process_predictions(prediction_results)`: Takes a list of YOLO prediction result objects, extracts mask and bounding box data, calculates areas, filters for "Cross Section" and "Vascular Bundles," and generates a pandas DataFrame of results.
    -   `_draw_research_style(...)`: Generates and saves an image with detected objects (Cross Section, Vascular Bundles) overlaid, suitable for research presentation.

## `src/app/api/test_converter.py`
-   **Purpose:** Test script for validating the `ImageConverter` module.
-   **Functionality:** Tests both single file and directory conversion of ND2 to PNG. Requires manual setup of test ND2 files.
-   **Status:** Manual test utility (not CI-gated runtime logic).

## `src/app/api/test_pipeline.py`
-   **Purpose:** Unit test for the full analysis pipeline.
-   **Functionality:** Tests the `run_full_pipeline` function end-to-end using a sample PNG image. Uses Python's `unittest` framework.
-   **Status:** Optional local smoke test; not required to run the ND2 -> PNG conversion flow.

This structure ensures a clear separation of concerns, making the application modular and maintainable.
