# Environment Setup Guide

This guide ensures a consistent development setup for the team using an Anaconda environment. Follow these steps to set up your environment before running the project code.

## 1. Install Anaconda
If Anaconda is not already installed, download it for your operating system from the [Anaconda Downloads page](https://www.anaconda.com/download). Follow the installer instructions to complete the setup.

Verify the installation by running:
```bash
conda --version
```

## 2. Create the Project Environment
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

## 3. Using the Environment
Before working on the project, always activate the environment:
```bash
conda activate daveBot-project-env
```

Run your Python code or Jupyter notebooks within this environment.

## 4. Updating the Environment
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
conda env update -f environment.yml --prune
```

## 5. Recreating the Environment (Optional)
If the environment encounters issues or you need a fresh start, remove and recreate it:
```bash
conda env remove -n daveBot-project-env
conda env create -f environment.yml
```