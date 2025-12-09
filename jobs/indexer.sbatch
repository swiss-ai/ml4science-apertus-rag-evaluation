#!/bin/bash
#SBATCH --job-name=warc-index
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=infra01

set -euo pipefail

# --- Paths on CSCS filesystem ---
JSONL_DIR=/capstor/scratch/cscs/$USER/ethz_jsonl
FAILED_INDEX=/capstor/scratch/cscs/$USER/failed_warcs_indexing.txt

# --- Elasticsearch config ---
export JSONL_DIR="$JSONL_DIR"
export ES_URL="https://es.swissai.cscs.ch"
export ES_INDEX_NAME="ml4science_ethz_webarchive"

# --- Embedding backend (OpenAI-compatible on CSCS, or Ollama if local) ---
# Example for SwissAI / CSCS:
export EMBED_PROVIDER="cscs"
export EMBED_MODEL="<model-name>"
export EMBED_BASE_URL="https://api.swissai.cscs.ch/v1"
export EMBED_API_KEY="<api-key>"

# If using local Ollama instead (not on CSCS):
# export EMBED_PROVIDER="ollama"
# export EMBED_MODEL="nomic-embed-text"

# Optional ES auth / TLS
# export ES_USER="elastic"
# export ES_PASSWORD="..."
# export ES_VERIFY_CERTS=true

# Indexing / chunking params
export BATCH_SIZE=32
export CHUNK_SIZE=1024
export CHUNK_OVERLAP=128
export FAILED_WARCS_INDEXING_FILE="$FAILED_INDEX"

# Optional: do not recreate index if it exists
# export NO_RECREATE_INDEX=1

export LOG_LEVEL=INFO
export LOG_FILE=/capstor/scratch/cscs/$USER/index_jsonl_${SLURM_JOB_ID}.log
export DEV_MODE=0   # no .env inside container

EDF_FILE=$SCRATCH/warc_tools.toml

# --- Run inside EDF container ---
set -x
srun --environment="$EDF_FILE" \
     warc-index
