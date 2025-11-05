# DaveBot Project

## Overview

Welcome to the DaveBot Project! This project is designed to create an intelligent chatbot for automating customer support. The chatbot will utilize natural language processing to understand user queries and provide relevant responses. This README provides instructions for setting up, running, and contributing to the project.

## Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Installation

To get started, ensure you have the following prerequisites installed:

- [Anaconda](https://www.anaconda.com/download) (for managing the project environment)
- [Git](https://git-scm.com/downloads) (for version control)

Clone the repository to your local machine:

```bash
git clone https://github.com/shashankreddy28/Dave-bot.git
cd Dave-bot
```

## Environment Setup

This project uses an Anaconda environment to ensure a consistent development setup. Follow these steps to set up the environment:

### 1. Install Anaconda

If Anaconda is not already installed, download it for your operating system from the [Anaconda Downloads page](https://www.anaconda.com/download). Follow the installer instructions to complete the setup.

Verify the installation by running:

```bash
conda --version
```

### 2. Create the Project Environment

Navigate to the project folder containing the `environment.yml` file:

```bash
cd path/to/project
```

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate daveBot-project-env
```

### 3. Using the Environment

Before working on the project, always activate the environment:

```bash
conda activate daveBot-project-env
```

Run your Python code or Jupyter notebooks within this environment.

### 4. Updating the Environment

If new dependencies are added to the project, update the `environment.yml` file. Add dependencies as follows:

- For packages available on conda-forge:

  ```yaml
  dependencies:
    - numpy
  ```

- For packages only available via pip:
  ```yaml
  - pip:
      - some-library
  ```

Update your local environment:

```bash
conda env update -f environment.yml
```

### 5. Recreating the Environment (Optional)

If the environment encounters issues or you need a fresh start, remove and recreate it:

```bash
conda env remove -n daveBot-project-env
conda env create -f environment.yml
```

## Usage

To run the project, ensure the environment is activated, then execute ..... (Placeholders for now):

```bash
python main.py
```

### Current Functionality

#### 1. Directory Setup

Sets up `data/` subfolder structure. Run the following in the root of the project:

```bash
python setup_directories.py
```

#### 2. Image Converters

Place any ND2 files you want to process in `data/nd2_images/input_images`. Next, move into the `image_converters/` folder:

```bash
cd src/image_converters
```

You have to run the TIFF converter first for now:

```bash
python tiff_converter.py
```

Then you can run the PNG converter:

```bash
python png_converter.py
```

All your processesed images can be found in `data/main_images/output_images/png_images/`.

## Project Structure

A brief overview of the project's directory structure:

```
Dave-bot/
├── data/                # Data files (not tracked in Git)
│   ├── main_images/
│   │   └── output_images/
│   │       ├── png_images/
│   │       └── tiff_images/
│   └── nd2_images/
│       └── input_images/
├── src/                 # Source code directory
├── environment.yml      # Anaconda environment configuration
├── .gitignore           # Information for git to ignore
├── main.py              # Main script to run the project
├── notebooks/           # Jupyter notebooks for experimentation
├── tests/               # Test scripts
└── README.md            # This file
```

### Data Folder Setup

```
Dave-bot/
├── data/
│   ├── main_images/
│   │   └── output_images/
│   │       ├── png_images/
│   │       └── tiff_images/
│   └── nd2_images/
│       └── input_images/
```

The `data/` directory (and all its subfolders) is not included in the repository because it is listed in `.gitignore`. This prevents large image files (ND2 and TIFF) from being accidentally committed. To ensure your image conversion scripts run correctly, you'll need to set up the exact image folder structure shown above.

#### Option 1: Run the Directory Setup Script

To automatically set up the folder structure in the correct format run the following from the root of the project folder:

```bash
python setup_directories.py
```

#### Option 2: Manually Create Folders

If you prefer to create the folders manually, please make sure you follow the structure shown above. The image conversion scripts depend on this organization to find input images and correctly place output files.
