#!/bin/bash
#SBATCH --job-name=yolo8seg_test
#SBATCH --account=heuschele_project
#SBATCH --output=yolo8seg_test_%j.out
#SBATCH --error=yolo8seg_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Load miniconda
module load miniconda

# Initialize conda
source /software/el9/apps/miniconda/24.7.1-2/etc/profile.d/conda.sh

# Remove old environment if it exists
conda env remove -n yolo_env -y 2>/dev/null || true

# Create fresh environment with Python 3.10 and required packages
echo "Creating conda environment with Python 3.10..."
conda create -n yolo_env python=3.10 -y

# Activate environment
conda activate yolo_env

# Install Ultralytics and dependencies
echo "Installing Ultralytics and dependencies..."
pip install ultralytics torch torchvision

# Print info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"
echo "Python version:"
python --version
echo "Python location:"
which python

# Verify packages
echo ""
echo "Verifying installations:"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"

# Run the training script
echo ""
echo "========================================"
echo "Starting YOLOv8 segmentation training..."
echo "========================================"
python train_yolov8_seg.py

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"

# List generated model files
echo ""
echo "Generated files in runs directory:"
ls -lh runs/segment/*/weights/*.pt 2>/dev/null || echo "No model files found"

conda deactivate