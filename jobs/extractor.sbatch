#!/bin/bash
#SBATCH --job-name=warc-extractor
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account="<account>"
#SBATCH --array=0-3

set -euo pipefail

# --- Paths on CSCS filesystem (outside container) ---
WARC_INPUT_DIR=/iopsstor/scratch/cscs/rashitig/19945
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/warc_ethz_jsonl
SEEDS_XLSX=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/urls/2025-12-01_19945_topics.xlsx
FAILED_WARCS=/capstor/scratch/cscs/$USER/failed_warcs_ethz_extraction.txt
EXTRACT_MODE=all

mkdir -p "$OUTPUT_DIR"

# --- Env vars that warc_to_jsonl.py expects ---
export WARC_INPUT_DIR="$WARC_INPUT_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"
export SEEDS_XLSX="$SEEDS_XLSX"
export FAILED_WARCS_FILE="$FAILED_WARCS"

# Optional sharding: single shard for now
export SHARD_COUNT=4
export SHARD_INDEX=$SLURM_ARRAY_TASK_ID

export LOG_LEVEL=INFO
export LOG_FILE=/capstor/scratch/cscs/$USER/warc_to_jsonl_${SLURM_JOB_ID}.log
export DEV_MODE=0   # no .env in container

# Path where EDF lives
EDF_FILE=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/jobs/warc_tools.toml

# --- Run inside the container environment ---
set -x
srun --environment="$EDF_FILE" \
     python -m warc_tools.extractor.cli
