#!/bin/bash
#SBATCH --job-name=warc-extract
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=<your_project>

set -euo pipefail

# --- Paths on CSCS filesystem (outside container) ---
WARC_DIR=/iopsstor/scratch/cscs/rashitig/19945
JSONL_DIR=/capstor/scratch/cscs/$USER/ethz_jsonl
SEEDS_XLSX=/capstor/scratch/cscs/$USER/ethz_seeds/19945_seeds.xlsx
FAILED_WARCS=/capstor/scratch/cscs/$USER/failed_warcs_extraction.txt

mkdir -p "$JSONL_DIR"

# --- Env vars that warc_to_jsonl.py expects ---
# Note: because EDF mounts /capstor and /iopsstor, these paths are the same inside the container.
export WARC_INPUT_DIR="$WARC_DIR"
export JSONL_DIR="$JSONL_DIR"
export SEEDS_XLSX="$SEEDS_XLSX"
export FAILED_WARCS_FILE="$FAILED_WARCS"

# Optional sharding: single shard for now
export SHARD_COUNT=1
export SHARD_INDEX=0

export LOG_LEVEL=INFO
export LOG_FILE=/capstor/scratch/cscs/$USER/warc_to_jsonl_${SLURM_JOB_ID}.log
export DEV_MODE=0   # no .env in container

# Path where EDF lives
EDF_FILE=$SCRATCH/ethz-warc/ethz-warc-extractor.toml

# --- Run inside the container environment ---
set -x
srun --environment="$EDF_FILE" \
     python warc_to_jsonl.py
