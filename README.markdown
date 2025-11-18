# Alfalfa Stem Object Detection (Dave Bot)

## Overview

This project provides a complete pipeline for analyzing microscopy images of alfalfa stems. It features a web-based user interface built with Gradio for easy interaction, allowing users to upload images, run an object detection model to identify key structures (cross-sections and vascular bundles), and receive detailed, research-ready outputs.

The core of this project is a YOLOv8 segmentation model trained to detect and segment vascular bundles in alfalfa stem . The repository includes scripts for data downloading, model training, and the main Gradio application.

## Features

-   **Gradio Web Interface:** An intuitive UI for easy image analysis without needing to run complex scripts manually.
-   **Dual Pipelines:**
    -   **Full Analysis Pipeline:** Converts images, runs the YOLOv8 model, post-processes the results to calculate areas in microns, and generates CSV reports and research-style overlay images.
    -   **Simple Image Converter:** A utility to convert `.nd2` microscopy files to `.png` format.
-   **Flexible File Handling:** Accepts `.nd2`, `.png`, and `.zip` files.
-   **Modular and Extensible:** The code is structured with a clear separation of concerns, making it easy to modify or extend.
-   **Reproducible Training:** Includes scripts and configuration for training the YOLOv8 model on a custom dataset.

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
│   └── ...
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
    *Note: The environment name is defined inside the `environment.yml` file.*

3.  **Verify the Setup:**
    If Anaconda was just installed, you can verify the installation by running `conda --version`.

### 4. Running the Gradio Application

The main application is the Gradio web interface.

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

### 3. Data Downloader

The `src/data_downloader` contains scripts to automate downloading and preparing data from Box.

-   **Configuration:** Add your Box API credentials to a `.env` file in the `src/data_downloader` directory.
-   **Usage:** Run `run_downloader.py` to start the process. See `downloader_documentation.md` for more details.

### 4. Model Training

The `model_training` directory contains the necessary files to train the YOLOv8 segmentation model.

-   **Dataset:** The training script expects a `data.yaml` file that defines the dataset paths and classes, following the YOLO format.
-   **Training:** The `train_yolov8_seg.py` script, orchestrated by `run.sh` in an HPC(High Performance Computer) environment, loads a base YOLOv8 model and trains it on the custom dataset. You can adapt `run.sh` for your local environment or run the python script directly.

## Configuration

-   **Model Selection:** To change the model used by the Gradio application, update the `DEFAULT_MODEL_FILENAME` variable in `src/app/main.py`.
-   **Post-Processing:** The pixel-to-micron ratio and other settings can be adjusted in `src/app/api/post_processor.py`.