#!/bin/bash
#SBATCH --job-name=warc-index
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account="<account>"
#SBATCH --array=0-3

set -euo pipefail

# --- Paths on CSCS filesystem ---
export JSONL_DIR=/iopsstor/scratch/cscs/$USER/warc_ethz_jsonl_filtered_ethz_only
export FAILED_WARCS_INDEXING_FILE=/iopsstor/scratch/cscs/$USER/ml4science_ethz_only_failed_indexing.txt

# Map array task -> year
YEARS_LIST=(2025 2024 2023 2022)
export YEARS="${YEARS_LIST[$SLURM_ARRAY_TASK_ID]}"
export MODALITIES=html,pdf

# --- Elasticsearch config ---
export ES_URL="https://es.swissai.cscs.ch"
export ES_INDEX_NAME="ml4science_ethz_19945_filtered"
export ES_VERIFY_CERTS=true
export NO_RECREATE_INDEX=1

# Indexing behavior
export BATCH_SIZE=16
export CHUNK_SIZE=768
export CHUNK_OVERLAP=16

# Bulk safety (anti-413)
export ES_BULK_MAX_MB=5
export ES_BULK_ACTIONS=1000
export ES_USER="<account>"
export ES_PASSWORD="<your-password>"

# Optional text guard (disable = 0)
export MAX_TEXT_CHARS=0

# Progress + restart safety (per-year checkpoint file)
export SKIP_COMPLETED_FILES=1
export COMPLETED_FILES_FILE="/iopsstor/scratch/cscs/$USER/index_ethz_only_completed_${YEARS}.txt"
export PROGRESS_EVERY=200

export FAILED_WARCS_INDEXING_FILE="/iopsstor/scratch/cscs/$USER/index_ethz_only_failed_warcs_${YEARS}.txt"

# --- Embedding backend ---
# Example for SwissAI / CSCS:
export EMBED_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0-ml4science
export EMBED_BASE_URL="https://api.swissai.cscs.ch/v1"
export EMBED_API_KEY="<api-key>"

# Logging
export LOG_LEVEL=INFO

unset SSL_CERT_FILE

# Path where EDF lives
EDF_FILE=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/jobs/warc_tools.toml

# --- Run inside EDF container ---
set -x
srun --environment="$EDF_FILE" \
     warc-index
