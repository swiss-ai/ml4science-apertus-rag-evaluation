#!/bin/bash
#SBATCH --job-name=warc-dedup
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account="<account>"

set -euo pipefail

# --- Paths on CSCS filesystem (outside container) ---
PROJECT_ROOT=/iopsstor/scratch/cscs/$USER
DATA_ROOT=$PROJECT_ROOT/warc_ethz_jsonl
OUT_ROOT=$PROJECT_ROOT/warc_ethz_jsonl_filtered
DB_ROOT=$PROJECT_ROOT/db

mkdir -p "$OUT_ROOT"
mkdir -p "$DB_ROOT"

export DEDUP_INPUT_ROOT="$DATA_ROOT"
export DEDUP_OUTPUT_ROOT="$OUT_ROOT"
export DEDUP_DB_DIR="$DB_ROOT"
export DEDUP_MODE=all           # html | pdf | all
export LOG_LEVEL=INFO

# Path where EDF lives
EDF_FILE=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/jobs/warc_tools.toml

# --- Run inside the container environment ---
set -x
srun --environment="$EDF_FILE" \
     warc-dedup
