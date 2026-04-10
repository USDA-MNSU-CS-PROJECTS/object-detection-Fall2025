# Alfalfa Stem Object Detection (Dave Bot)

## Overview

This project provides a complete pipeline for analyzing microscopic images of alfalfa stems. It features a web-based user interface built with Gradio for easy interaction, allowing users to upload images, run an object detection model to identify key structures (cross-sections and vascular bundles), and receive detailed, research-ready outputs.

The core of this project is a YOLOv8 segmentation model trained to detect and segment vascular bundles in alfalfa stem . The repository includes scripts for data downloading, model training, the main Gradio application, and a portable packaging system for easy distribution to end users.

## Features

- **Gradio Web Interface:** An intuitive UI for easy image analysis without needing to run complex scripts manually.
- **Dual Pipelines:**
  - **Full Analysis Pipeline:** Converts images, runs the YOLOv8 model, post-processes the results to calculate areas in microns, and generates CSV reports and research-style overlay images.
  - **Simple Image Converter:** A utility to convert `.nd2` microscopy files to `.png` format.
- **Flexible File Handling:** Accepts `.nd2`, `.png`, and `.zip` files.
- **Modular and Extensible:** The code is structured with a clear separation of concerns, making it easy to modify or extend.
- **Reproducible Training:** Includes scripts and configuration for training the YOLOv8 model on a custom dataset.
- **Portable Distribution:** Build a standalone package that end users can run without installing Python or any dependencies.

## Project Structure

```
├── data/                  # Directory for input images
├── model_training/        # Scripts and configuration for model training
│   ├── data.yaml          # Dataset configuration for YOLO
│   ├── run.sh             # SLURM script for HPC training
│   └── train_yolov8_seg.py # Main training script
├── sample_trained_models/ # Pre-trained model weights
├── src/
│   ├── app/               # Gradio application source code
│   │   ├── main.py        # Main Gradio app entry point
│   │   └── api/           # Backend modules for the app
│   │       ├── converter.py
│   │       ├── predictor.py
│   │       └── post_processor.py
│   ├── data_downloader/   # Scripts for downloading data from Box
│   └── image_converters/  # Scripts for converting .nd2 files to .png files
│   └── model_inference/   # Scripts for running the trained YOLOv8 model
│   └── post_processer/    # Scripts for post-processing the results from model inference
├── create_portable_package.bat  # Build script for creating distributable package
├── .gitignore
├── environment.yml        # Conda environment definition
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

- **Model Selection:** To change the model used by the Gradio application, update the `DEFAULT_MODEL_FILENAME` variable in `src/app/main.py`. For the portable distribution, the model path is automatically configured to use `models/best.pt` relative to the application directory.
- **Post-Processing:** The pixel-to-micron ratio and other settings can be adjusted in `src/app/api/post_processor.py`.

## Distribution Package Details

### What's Included

The portable package created by `create_portable_package.bat` includes:

- Complete Python 3.9 environment with all dependencies
- The Gradio application and all backend modules
- Pre-trained YOLOv8 model (`best.pt`)
- User-friendly launcher (`START_TOOL.bat`)
- Comprehensive user documentation (`README.txt`)

### System Requirements for End Users

- Windows 10 or later
- 8GB RAM minimum (16GB recommended for processing large batches)
- 2GB free disk space
- No Python installation or technical knowledge required

### Updating the Model

To update the model in an existing portable package:

1. Replace `portable_package/models/best.pt` with the new model file
2. Re-compress the folder for distribution
3. No rebuild necessary!

### Performance Notes

The portable distribution uses CPU-only PyTorch for maximum compatibility across different systems. Processing will be slower than GPU-accelerated inference but works on any Windows computer without requiring specialized hardware.
