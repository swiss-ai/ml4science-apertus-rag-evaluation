# src/warc_tools/indexer/cli.py
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .utils import require_env, setup_logging
from .indexer import IndexerConfig, run_indexing


def main() -> None:
    """
    CLI entrypoint for JSONL → Elasticsearch indexing.

    Reads configuration from environment variables, sets up logging,
    and calls run_indexing().
    """
    dev_mode = os.getenv("DEV_MODE", "0").lower() in ("1", "true", "yes")
    if dev_mode:
        load_dotenv()

    # Required env vars
    jsonl_dir_str = require_env("JSONL_DIR")
    es_url = require_env("ES_URL")
    index_name = require_env("ES_INDEX_NAME")
    batch_size = int(require_env("BATCH_SIZE"))
    chunk_size = int(require_env("CHUNK_SIZE"))
    chunk_overlap = int(require_env("CHUNK_OVERLAP"))
    failed_warcs_str = require_env("FAILED_WARCS_INDEXING_FILE")

    # Optional env vars
    es_user = os.getenv("ES_USER") or None
    es_password = os.getenv("ES_PASSWORD") or None
    es_verify_certs = os.getenv("ES_VERIFY_CERTS", "false").lower() in ("1", "true", "yes")
    no_recreate_index = os.getenv("NO_RECREATE_INDEX", "0").lower() in ("1", "true", "yes")

    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE") or None

    logger = setup_logging(log_level, log_file)

    config = IndexerConfig(
        jsonl_dir=Path(jsonl_dir_str),
        es_url=es_url,
        index_name=index_name,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        failed_warcs_file=Path(failed_warcs_str),
        es_user=es_user,
        es_password=es_password,
        es_verify_certs=es_verify_certs,
        no_recreate_index=no_recreate_index,
    )

    try:
        run_indexing(config, logger)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
