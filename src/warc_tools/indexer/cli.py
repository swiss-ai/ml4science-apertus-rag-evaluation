# src/warc_tools/indexer/cli.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .utils import require_env, setup_logging
from .config import IndexerConfig
from .indexer import run_indexing

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main() -> int:
    load_dotenv()

    logger = setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE") or None,
    )

    config = IndexerConfig(
        jsonl_dir=Path(require_env("JSONL_DIR")),
        es_url=require_env("ES_URL"),
        index_name=require_env("ES_INDEX_NAME"),
        batch_size=int(require_env("BATCH_SIZE")),
        chunk_size=int(require_env("CHUNK_SIZE")),
        chunk_overlap=int(require_env("CHUNK_OVERLAP")),
        failed_warcs_file=Path(require_env("FAILED_WARCS_INDEXING_FILE")),
        es_user=os.getenv("ES_USER") or None,
        es_password=os.getenv("ES_PASSWORD") or None,
        es_verify_certs=os.getenv("ES_VERIFY_CERTS", "false").lower() in ("1", "true", "yes"),
        no_recreate_index=os.getenv("NO_RECREATE_INDEX", "0").lower() in ("1", "true", "yes"),
    )

    try:
        run_indexing(config, logger)
        return 0
    except Exception:
        logger.exception("Indexing failed")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
