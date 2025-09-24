

🛠️ Environment Setup Guide

To ensure everyone on the team has the same development setup, we use an Anaconda environment. Follow these steps before running the code.

1. Install Anaconda (if not already installed)

  Download Anaconda for your operating system:
   [Anaconda Downloads](https://www.anaconda.com/download/success)
  Follow the installer instructions.
  
  Verify the installation: 
  ```bash
conda --version
```

2. Create the Project Environment

  Navigate to the project folder (where environment.yml is located).
  
  `cd path/to/project`
  
  Create the environment:
  
  `conda env create -f environment.yml`
  
  
  Activate the environment:
  
  `conda activate daveBot-project-env`

3. Using the Environment

  Every time you work on the project, activate the environment:
  
  `conda activate nd2-project-env`


  Run your Python code or Jupyter notebooks inside this environment.

4. Updating the Environment

  If we add new dependencies to the project:
  
  Add the new library to environment.yml.
  
  If available on conda-forge:
  
    dependencies:
      - numpy
  
  
  If available only via pip:
  
    - pip:
        - some-library


  Update your local environment:
  
  `conda env update -f environment.yml --prune`

5. Recreating the Environment (Optional)

  If something breaks or you want a fresh start:
  
  `conda env remove -n nd2-project-env`
  `conda env create -f environment.yml`
