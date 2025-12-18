#!/bin/bash
#SBATCH --job-name=warc-rag
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --account="<account>"

set -euo pipefail

# --- Check that a query was provided ---
if [ -z "${RAG_QUERY:-}" ]; then
  echo "ERROR: RAG_QUERY environment variable not set." >&2
  echo "Usage: RAG_QUERY=\"your question\" sbatch warc_rag_query.sbatch" >&2
  exit 1
fi

# --- Elasticsearch configuration ---
export ES_URL="https://es.swissai.cscs.ch"
export ES_INDEX_NAME="ml4science_ethz_19945_filtered"

# ES basic auth (required in your setup)
export ES_USER="<account>"           # or your ES user
export ES_PASSWORD="<your-password>"
export ES_VERIFY_CERTS=false        # true if proper CA, false for self-signed

# --- Embedding backend (must match what you used for indexing) ---
# Example: OpenAI-compatible (SwissAI / CSCS)
export EMBED_PROVIDER="cscs"        # or "ollama" / "hf"
export EMBED_MODEL="Snowflake/snowflake-arctic-embed-l-v2.0-ml4science"
export EMBED_BASE_URL="https://api.swissai.cscs.ch/v1"
export EMBED_API_KEY="<embed-api-key>"

# --- LLM backend for answer generation ---
# Example: same SwissAI / CSCS endpoint
export LLM_PROVIDER="cscs"          # or "ollama" / "hf"
export LLM_MODEL="Qwen/Qwen3-8B"
export LLM_BASE_URL="https://api.swissai.cscs.ch/v1"
export LLM_API_KEY="<llm-api-key>"

# Tuning for the LLM
export LLM_TEMPERATURE=0.1
export LLM_MAX_TOKENS=1024

# --- RAG parameters ---
export RAG_TOP_K=5

# --- Logging ---
export LOG_LEVEL=INFO
export LOG_FILE=/capstor/scratch/cscs/$USER/warc_rag_${SLURM_JOB_ID}.log
export DEV_MODE=0   # no .env inside the container

EDF_FILE=/capstor/scratch/cscs/$USER/ml4science-apertus-rag-evaluation/jobs/warc_tools.toml

echo "Running RAG query: '$RAG_QUERY'"

set -x
srun --environment="$EDF_FILE" \
     warc-rag "$RAG_QUERY"
