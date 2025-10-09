import os
import subprocess

# Activate conda environment and run the script
conda_env = "daveBot-project-env"
script_path = os.path.join(os.path.dirname(__file__), "downloader.py")

subprocess.run(["/opt/anaconda3/bin/conda", "run", "-n", conda_env, "python", script_path])
