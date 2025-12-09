# src/extractor/cli.py
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .utils import require_env, setup_logging
from .warc_extractor import ExtractorConfig, run_extraction


def main() -> None:
    """
    CLI entrypoint for WARC → JSONL extraction.

    Reads configuration from environment variables, sets up logging,
    and calls run_extraction().
    """
    # For local dev you can set DEV_MODE=1 and use a .env file.
    dev_mode = os.getenv("DEV_MODE", "0").lower() in ("1", "true", "yes")
    if dev_mode:
        load_dotenv()

    # Required env vars
    input_dir_str = require_env("WARC_INPUT_DIR")
    output_dir_str = require_env("JSONL_DIR")
    seeds_xlsx_str = require_env("SEEDS_XLSX")
    failed_warcs_str = require_env("FAILED_WARCS_FILE")

    # Optional env vars
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE") or None
    shard_idx = int(os.getenv("SHARD_INDEX", "0"))
    shard_cnt = int(os.getenv("SHARD_COUNT", "1"))

    logger = setup_logging(log_level, log_file)

    config = ExtractorConfig(
        input_dir=Path(input_dir_str),
        output_dir=Path(output_dir_str),
        seeds_xlsx=Path(seeds_xlsx_str),
        failed_warcs_file=Path(failed_warcs_str),
        shard_index=shard_idx,
        shard_count=shard_cnt,
    )

    try:
        run_extraction(config, logger)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
