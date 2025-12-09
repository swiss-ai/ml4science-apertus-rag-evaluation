#!/bin/bash
#SBATCH --job-name=warc_extract
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=warc_extract_%j.out
#SBATCH --error=warc_extract_%j.err
#SBATCH --account=infra01

# Load conda
echo "Loading conda..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_warc

# Check if conda activated
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'env_warc'"
    exit 1
fi

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Navigate to project directory
echo "Changing to project directory..."
cd /users/alouah/serving/ethz_webarchive || {
    echo "ERROR: Failed to change to project directory"
    exit 1
}

echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la | head -10

# Check if script exists (WSL filename run_warc.py)
if [ ! -f "run_warc.py" ]; then
    echo "ERROR: run_warc.py not found in $(pwd)"
    exit 1
fi

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

# Run extraction
echo ""
echo "Starting WARC extraction at $(date)"
echo "=========================================="
python -u run_warc.py 2>&1
EXIT_CODE=$?
echo "=========================================="
echo "Finished WARC extraction at $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Script exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

