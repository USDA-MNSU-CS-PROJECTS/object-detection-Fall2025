# Alfalfa Stem Object Detection (Dave Bot)

## Overview

This project provides a complete pipeline for analyzing microscopic images of alfalfa stems. It features a web-based user interface built with Gradio for easy interaction, allowing users to upload images, run **two** YOLOv8 segmentation models (Casparian + Epidermis, and vascular bundles), post-process ring/Casparian noise, and receive detailed, research-ready outputs (CSV, overlays, optional debug dumps).

The repository includes scripts for data downloading, model training, the main Gradio application, and a portable packaging system for easy distribution to end users.

## Features

- **Gradio Web Interface:** An intuitive UI for easy image analysis without needing to run complex scripts manually.
- **Dual Pipelines:**
  - **Full Analysis Pipeline:** Converts images, runs model A (Casparian + Epidermis) and model B (vascular bundles), applies noise handling in ROIs, computes geometry in microns, and generates CSV reports and overlay PNGs.
  - **Simple Image Converter:** A utility to convert `.nd2` microscopy files to `.png` format.
- **Flexible File Handling:** Accepts `.nd2`, `.png`, `.jpg`, `.jpeg`, and `.zip` files.
- **Modular and Extensible:** The code is structured with a clear separation of concerns, making it easy to modify or extend.
- **Reproducible Training:** Includes scripts and configuration for training the YOLOv8 model on a custom dataset.
- **Portable Distribution:** Build a standalone package that end users can run without installing Python or any dependencies.

## Project Structure

```
├── data/                       # Directory for input images (gitignored by default)
├── debug_output/               # Optional debug logs/JSON from runs (gitignored)
├── model_training/             # Scripts and configuration for model training
│   ├── data.yaml               # Dataset configuration for YOLO
│   ├── train_yolov8_seg.py     # Main training script
│   └── model_training_readme.md
├── sample_trained_models/      # Pre-trained weights (add locally; see Configuration)
├── src/
│   ├── app/                    # Gradio application
│   │   ├── app.py              # Run the UI (recommended entry point)
│   │   ├── main.py             # UI + pipeline implementation
│   │   ├── config/             # inference_constants, noise_profiles_app
│   │   └── api/
│   │       ├── converter.py
│   │       ├── predictor.py
│   │       ├── dual_stem_pipeline.py  # Default dual-model post-process + viz
│   │       ├── stem_metrics.py
│   │       ├── yolo_label_export.py
│   │       ├── filename_metadata.py
│   │       ├── post_processor.py        # Legacy single-model viz (not default UI path)
│   │       └── test_pipeline.py         # Integration smoke test
│   ├── third_party/
│   │   └── noise_deletion_clean/        # ROI noise detection (used by dual pipeline)
│   ├── data_downloader/        # Box downloader (optional)
│   ├── image_converters/       # .nd2 → .png helpers
│   └── model_inference/        # Standalone inference scripts / docs
├── setup_directories.py
├── create_portable_package.bat
├── environment.yml          # Conda environment definition
├── requirements.txt         # pip install -r requirements.txt (venv workflow)
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

You can set up the project with **either** Conda **or** a plain Python virtual environment + `pip`. Pick one — they install the same dependencies.

#### Option A: pip + venv (recommended for Windows clients via the Alfalfa Unified UI)

Requires Python 3.9 on `PATH` (gradio 3.50.2 — pinned in `requirements.txt` — is the only Gradio version that works reliably with the rest of this stack on Python 3.9).

From the repository root:

**Windows (cmd):**

```cmd
cd C:\Users\<you>\alfalfa-tools\object-detection-Fall2025
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch the UI (see [§4 Running the Gradio Application](#4-running-the-gradio-application)).

#### Option B: Conda

This project also ships a Conda environment file for a consistent development setup.

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

3.  **(Optional, legacy) Create Data Directories:**
    Run the setup script only if you intentionally use legacy/manual converter tests:

    ```bash
    python setup_directories.py
    ```

    The main Gradio workflow (`src/app/main.py` plus `src/app/app.py`) uses temporary runtime folders and does **not** require this step.  
    `setup_directories.py` is kept for older/manual local workflows (`converter_test_data`, legacy data layout).

4.  **Verify the Setup:**
    If Anaconda was just installed, you can verify the installation by running `conda --version`.

### 4. Running the Gradio Application

The main application is the Gradio web interface.

#### For Development and Testing

To run the application locally for development or testing purposes:

1.  **Ensure the environment is active** (either the venv from Option A or the Conda env from Option B).
2.  **Navigate to the application directory and start the UI:**

    **Windows (cmd):**

    ```cmd
    cd src\app
    python main.py
    ```

    **macOS / Linux:**

    ```bash
    cd src/app
    python main.py
    ```

    `main.py` is the canonical entry point (it sets up `sys.path` for the bundled `api/` and `third_party/` modules, then launches the same Gradio `app` that `app.py` wraps). `python app.py` also works and accepts `--port` / `--no-browser` flags.

3.  Open your web browser at the URL printed by Gradio. The default port is **7860**; the Alfalfa Unified UI launcher typically pins this tool to **7861** via the `PORT` environment variable.

Optional: set a port with `--port` (when launching via `app.py`), or with the `GRADIO_SERVER_PORT` / `PORT` environment variables (`PORT` is what the Alfalfa Unified UI uses).

### Supported upload formats

- `.nd2`
- `.png`
- `.jpg`
- `.jpeg`
- `.zip`

JPEG is supported, but PNG is preferred when you need tighter control over pixels, because JPEG uses lossy compression.

### Exported results layout (reference)

Each analysis run builds a workspace under a temporary folder: `input/`, `converted/`, `output/`. When you download `output_images.zip`, paths follow this structure:

- `input_images/` — copies of raster inputs passed to inference (basename preserved)
- `visualizations/` — overlay images (when enabled by configuration)
- `labels_generated/` — YOLO-style label text files
- `geometry_export/` — merged geometry text files
- `metric_debug_viz/` — optional per-metric debug PNGs when enabled in settings
- `debug/` — log and raw prediction JSON summaries (copied from project `debug_output/` when present)

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

The `src/data_downloader` folder contains a one-time legacy utility for downloading and preparing data from Box. It was written for a specific historical ingest path and uses hardcoded assumptions. [Get more information](src/data_downloader/downloader_documentation.md)

- **Configuration:** Add your Box API credentials to a `.env` file in the `src/data_downloader` directory.
- **Usage note:** `run_downloader.py` currently calls Conda using a hardcoded macOS path (`/opt/anaconda3/bin/conda`). On Windows/Linux, run `downloader.py` manually in your environment or update the launcher script first.

### 6. Model Training

The `model_training` directory contains the necessary files to train the YOLOv8 segmentation model.

- **Dataset:** The training script expects a `data.yaml` file that defines the dataset paths and classes, following the YOLO format.
- **Training:** Run `train_yolov8_seg.py` from the `model_training` directory (after editing `data.yaml` and script parameters as needed). For SLURM or other schedulers, wrap the same command in your own job script at the cluster. Details: [model_training/model_training_readme.md](model_training/model_training_readme.md).

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
- **Important compatibility note:** `create_portable_package.bat` is currently a **legacy single-model packager** (`models/best.pt` flow). The current app runtime in `src/app/main.py` is **dual-model** and expects `casparian_epidermis.pt` + `vascular_bundles.pt`. Treat the generated portable package as legacy unless you update launcher/model path logic for dual-model mode.
- User-friendly launcher (`START_TOOL.bat`)
- Comprehensive user documentation (`README.txt`)

### System Requirements for End Users

- Windows 10 or later
- 8GB RAM minimum (16GB recommended for processing large batches)
- 2GB free disk space
- No Python installation or technical knowledge required

### Updating the Model

To update weights in an existing portable package, ensure the launcher and app paths match your packaging mode:

- Legacy packager mode: replace `portable_package/models/best.pt`.
- Current app dual-model mode: provide both model files and align launcher/app paths with `inference_constants.py` (`casparian_epidermis.pt`, `vascular_bundles.pt`).

### Performance Notes

The portable distribution uses CPU-only PyTorch for maximum compatibility across different systems. Processing will be slower than GPU-accelerated inference but works on any Windows computer without requiring specialized hardware.
