#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account="<account>"

set -euo pipefail

# CSCS OpenAI-compatible endpoint
export CSCS_BASE_URL="https://api.swissai.cscs.ch/v1"
export CSCS_API_KEY="<api-key>"

# Baseline model configuration
export BASELINE_MODEL="Qwen/Qwen3-8B"
export BASELINE_TEMPERATURE="0.0"
export BASELINE_SLEEP_SECONDS="1.5"
export BASELINE_SKIP_EXISTING="1"

# Optional system prompt
export BASELINE_SYSTEM_PROMPT="You are a helpful assistant answering questions clearly and concisely."

# Logging
export LOG_LEVEL="INFO"

INPUT_XLSX=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/evaluation/evaluation-qwen-out.xlsx
OUTPUT_XLSX=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/evaluation/evaluation-qwen-apertus-out.xlsx

# Path where EDF lives
EDF_FILE=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/jobs/warc_tools.toml

unset SSL_CERT_FILE

# --- Run inside the container environment ---
set -x
srun --environment="$EDF_FILE" \
     python -m warc_tools.baseline.cli "$INPUT_XLSX" "$OUTPUT_XLSX" "$BASELINE_MODEL"
