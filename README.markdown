# Alfalfa Stem Object Detection (Dave Bot)

## Overview

This project provides a complete pipeline for analyzing microscopic images of alfalfa stems. It features a web-based user interface built with Gradio for easy interaction, allowing users to upload images, run **two** YOLOv8 segmentation models (Casparian + Epidermis, and vascular bundles), post-process ring/Casparian noise, and receive detailed, research-ready outputs (CSV, overlays, optional debug dumps).

The repository includes scripts for data downloading, model training, the main Gradio application, and a portable packaging system for easy distribution to end users.

## Features

- **Gradio Web Interface:** An intuitive UI for easy image analysis without needing to run complex scripts manually.
- **Dual Pipelines:**
  - **Full Analysis Pipeline:** Converts images, runs model A (Casparian + Epidermis) and model B (vascular bundles), applies noise handling in ROIs, computes geometry in microns, and generates CSV reports and overlay PNGs.
  - **Simple Image Converter:** A utility to convert `.nd2` microscopy files to `.png` format.
- **Flexible File Handling:** Accepts `.nd2`, `.png`, and `.zip` files.
- **Modular and Extensible:** The code is structured with a clear separation of concerns, making it easy to modify or extend.
- **Reproducible Training:** Includes scripts and configuration for training the YOLOv8 model on a custom dataset.
- **Portable Distribution:** Build a standalone package that end users can run without installing Python or any dependencies.

## Project Structure

```
├── data/                       # Directory for input images (gitignored by default)
├── debug_output/               # Optional debug logs/JSON from runs (gitignored)
├── model_training/             # Scripts and configuration for model training
│   ├── data.yaml               # Dataset configuration for YOLO
│   ├── run.sh                  # SLURM script for HPC training
│   └── train_yolov8_seg.py     # Main training script
├── sample_trained_models/      # Pre-trained weights (add locally; see Configuration)
├── src/
│   ├── app/                    # Gradio application
│   │   ├── main.py             # Entry point
│   │   ├── config/             # inference_constants, noise_profiles_app
│   │   └── api/
│   │       ├── converter.py
│   │       ├── predictor.py
│   │       ├── dual_stem_pipeline.py  # Default dual-model post-process + viz
│   │       ├── stem_metrics.py
│   │       ├── yolo_label_export.py
│   │       ├── post_processor.py        # Legacy single-model viz (not default UI path)
│   │       └── test_pipeline.py         # Integration smoke test
│   ├── third_party/
│   │   └── noise_deletion_clean/        # ROI noise detection (used by dual pipeline)
│   ├── data_downloader/        # Box downloader (optional)
│   ├── image_converters/       # .nd2 → .png helpers
│   └── model_inference/        # Standalone inference scripts / docs
├── setup_directories.py
├── create_portable_package.bat
├── environment.yml
└── README.markdown
```

## Installation and Setup

### 1. Prerequisites

- [Git](https://git-scm.com/downloads) (for version control)
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (for managing the project environment)

### 2. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/shashankreddy28/Dave-bot.git
cd Dave-bot
```

### 3. Environment Setup

This project uses a Conda environment to ensure a consistent development setup.

1.  **Create the Conda Environment:**
    Navigate to the project root directory (where `environment.yml` is located) and run the following command to create the environment:

    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment:**
    Before running the application, you must activate the environment:

    ```bash
    conda activate daveBot-project-env
    ```

    _Note: The environment name is defined inside the `environment.yml` file._

3.  **Create Data Directories:**
    Run the setup script to create the necessary directory structure for data storage and testing:

    ```bash
    python setup_directories.py
    ```

    This script creates directories for input images, output images, and test data used during development and testing. The directories are hardcoded in the test scripts.

4.  **Verify the Setup:**
    If Anaconda was just installed, you can verify the installation by running `conda --version`.

### 4. Running the Gradio Application

The main application is the Gradio web interface.

#### For Development and Testing

To run the application locally for development or testing purposes:

1.  **Ensure the environment is active.**
2.  **Navigate to the application directory:**
    ```bash
    cd src/app
    ```
3.  **Run the app:**
    ```bash
    python main.py
    ```
4.  Open your web browser and go to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

#### For End User Distribution

To create a portable package that end users can run without installing Python:

1.  **Ensure you're in the project root directory** and the conda environment is active:

    ```bash
    conda activate daveBot-project-env
    ```

2.  **Run the portable package builder:**

    ```bash
    create_portable_package.bat
    ```

    This script will:

    - Create a new isolated Python environment
    - Install all necessary dependencies (using CPU-only PyTorch for broad compatibility)
    - Copy the application files and trained model
    - Package everything into a `portable_package` folder
    - Generate user-friendly launcher scripts and documentation

    **Note:** This process takes 10-30 minutes and creates a ~3-4.5GB folder.

3.  **Test the portable package:**

    ```bash
    cd portable_package
    START_TOOL.bat
    ```

4.  **Distribute to end users:**
    - Compress the `portable_package` folder using 7-Zip or similar (Windows built-in compression is very slow for large folders)
    - Share the compressed file via OneDrive, Google Drive, or other file sharing service
    - End users simply unzip and run `START_TOOL.bat` - no installation required!

### 5. Data Downloader

The `src/data_downloader` contains scripts to automate downloading and preparing data from Box. This was a one time use code that uses box developer console and API keys to download the data from Box. [Get more information](src/data_downloader/downloader_documentation.md)

- **Configuration:** Add your Box API credentials to a `.env` file in the `src/data_downloader` directory.
- **Usage:** Run `run_downloader.py` to start the process.

### 6. Model Training

The `model_training` directory contains the necessary files to train the YOLOv8 segmentation model.

- **Dataset:** The training script expects a `data.yaml` file that defines the dataset paths and classes, following the YOLO format.
- **Training:** The `train_yolov8_seg.py` script, orchestrated by `run.sh` in an HPC(High Performance Computer) environment, loads a base YOLOv8 model and trains it on the custom dataset. You can adapt `run.sh` for your local environment or run the python script directly.

## Configuration

- **Models (Gradio):** Two weights in `sample_trained_models/`: `casparian_epidermis.pt` (Casparian + Epidermis) and `vascular_bundles.pt` (VB). Filenames and class names are set in `src/app/config/inference_constants.py`. Noise thresholds (PG/RR × ring/casp) — `src/app/config/noise_profiles_app.py`.
- **Geometry metrics:** `src/app/api/stem_metrics.py`. Legacy single-model post-process remains in `src/app/api/post_processor.py` (not used by the default Gradio path).
- **Bundled noise helpers:** `src/third_party/noise_deletion_clean` only ships `detection`, `masks`, and `profiles` (CLI/compare entry points were removed; Gradio is the supported path).

### Visualization colors (default dual pipeline)

Overlays written by `dual_stem_pipeline` use: **red** — epidermis contours; **deepskyblue** — selected main Casparian polygon; **green** (semi-transparent) — vascular bundles; **red-tinted semi-transparent** — detected noise mask over the image (not a YOLO class).

## Distribution Package Details

### What's Included

The portable package created by `create_portable_package.bat` includes:

- Complete Python 3.9 environment with all dependencies
- The Gradio application and all backend modules
- **Note:** The batch script copies `sample_trained_models/best.pt` to `portable_package/models/best.pt` if that file exists. The **current** `src/app/main.py` expects **two** weights under `sample_trained_models/` (`casparian_epidermis.pt` and `vascular_bundles.pt` per `inference_constants.py`). For development runs, use those two files; if you rely on the portable builder, copy or symlink both weights into the portable `models/` folder and align paths with your deployment (the generated launcher still references `models/best.pt` from older single-model flows).
- User-friendly launcher (`START_TOOL.bat`)
- Comprehensive user documentation (`README.txt`)

### System Requirements for End Users

- Windows 10 or later
- 8GB RAM minimum (16GB recommended for processing large batches)
- 2GB free disk space
- No Python installation or technical knowledge required

### Updating the Model

To update weights in an existing portable package, replace the files under `portable_package/models/` with your new `.pt` files and ensure `main.py` / launcher paths match the filenames you use. If you still use the generated `best.pt` layout from `create_portable_package.bat`, replace `portable_package/models/best.pt`; for the dual-model app, supply both model A and B weights consistently with `inference_constants.py`.

### Performance Notes

The portable distribution uses CPU-only PyTorch for maximum compatibility across different systems. Processing will be slower than GPU-accelerated inference but works on any Windows computer without requiring specialized hardware.
