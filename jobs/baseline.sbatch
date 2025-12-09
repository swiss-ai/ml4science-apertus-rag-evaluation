#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=infra01

set -euo pipefail

# CSCS OpenAI-compatible endpoint
export CSCS_BASE_URL="https://api.swissai.cscs.ch/v1"
export CSCS_API_KEY="<api-key>"

mkdir -p "$JSONL_DIR"

# Baseline model configuration
export BASELINE_MODEL="Qwen/Qwen3-8B"
export BASELINE_TEMPERATURE="0.0"
export BASELINE_SLEEP_SECONDS="1.5"
export BASELINE_SKIP_EXISTING="1"

# Optional system prompt
export BASELINE_SYSTEM_PROMPT="You are a helpful assistant answering questions clearly and concisely."

# Logging
export LOG_LEVEL="INFO"

# Path where EDF lives
EDF_FILE=$SCRATCH/warc_tools.toml

# --- Run inside the container environment ---
set -x
srun --environment="$EDF_FILE" \
     python warc_to_jsonl.py
