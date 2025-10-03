#!/bin/bash
#SBATCH --job-name=yolo8seg-test
#SBATCH --account=heuschele_project
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00       # 1 hour is plenty for a 10-image test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load necessary modules
module load python/3.10
module load cuda/12.1

# (Optional) activate your virtual environment
source ~/envs/yolo-env/bin/activate

cd ~/project   # replace with your project path on SciNet

# Install Ultralytics if not already in the environment
pip install --quiet ultralytics

# Run the training script
python train_yolov8_seg.py
