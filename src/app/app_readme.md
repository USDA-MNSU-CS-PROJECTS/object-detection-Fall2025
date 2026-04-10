# Application Code Overview

This document provides an overview of the core Python code that powers the Alfalfa Stem Object Detection Gradio application. The application is structured into a main Gradio interface (`main.py`) and several API modules (`converter.py`, `predictor.py`, `post_processor.py`) responsible for specific tasks.

## `src/app/main.py`
-   **Purpose:** Serves as the main entry point for the Gradio application. It defines the user interface, handles file uploads, orchestrates the processing pipeline, and displays results.
-   **Key Functions:**
    -   `run_conversion(files)`: Handles the "Simple Image Converter" pipeline, converting ND2 files to PNG and zipping them for download.
    -   `run_full_pipeline(files, progress=gr.Progress())`: Manages the "Full Analysis Pipeline," which includes image conversion, model prediction, post-processing, and generating CSV reports and overlay images. The confidence threshold is configured via `DEFAULT_CONF_THRESHOLD` constant.
-   **Model Configuration:** The `DEFAULT_MODEL_FILENAME` and `DEFAULT_CONF_THRESHOLD` variables at the top of this file allow easy switching of the trained YOLO model and confidence threshold used for predictions.
-   **File Handling:** Accepts `.nd2`, `.png`, and `.zip` files. For `.zip` files, it extracts only `.nd2` and `.png` files, ignoring others and macOS metadata files (e.g., `._*`).

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

## `src/app/api/post_processor.py`
-   **Purpose:** Processes the raw prediction results from the YOLO model, performs calculations (e.g., area in microns), and generates research-style visualizations.
-   **Key Functions:**
    -   `PostProcessor.__init__(output_dir, pixel_to_micron_ratio)`: Initializes the post-processor, setting up output directories and the pixel-to-micron conversion ratio.
    -   `PostProcessor.process_predictions(prediction_results)`: Takes a list of YOLO prediction result objects, extracts mask and bounding box data, calculates areas, filters for "Cross Section" and "Vascular Bundles," and generates a pandas DataFrame of results.
    -   `_draw_research_style(...)`: Generates and saves an image with detected objects (Cross Section, Vascular Bundles) overlaid, suitable for research presentation.

## `src/app/api/test_converter.py`
-   **Purpose:** Test script for validating the `ImageConverter` module.
-   **Functionality:** Tests both single file and directory conversion of ND2 to PNG. Requires manual setup of test ND2 files.

## `src/app/api/test_pipeline.py`
-   **Purpose:** Unit test for the full analysis pipeline.
-   **Functionality:** Tests the `run_full_pipeline` function end-to-end using a sample PNG image. Uses Python's `unittest` framework.

This structure ensures a clear separation of concerns, making the application modular and maintainable.
