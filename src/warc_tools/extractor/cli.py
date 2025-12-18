"""
WARC Extractor CLI Module

Command-line interface for extracting content from WARC files.
Reads configuration from environment variables and processes WARC files
to extract HTML and/or PDF content.

Environment Variables:
    - WARC_INPUT_DIR: Directory containing WARC files
    - OUTPUT_DIR: Output directory for extracted content
    - SEEDS_XLSX: Excel file with seed URLs
    - EXTRACT_MODE: "html", "pdf", or "all"
"""
from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv

from .config import ExtractorConfig
from .utils import require_env_path, setup_logging
from .warc_extractor import run_extraction

ExtractMode = Literal["html", "pdf", "all"]
_VALID_MODES = {"html", "pdf", "all"}


def _parse_extract_mode(raw: str | None) -> tuple[bool, bool]:
    mode = (raw or "all").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid EXTRACT_MODE={mode!r}")
    return mode in {"html", "all"}, mode in {"pdf", "all"}

def _parse_sharding() -> tuple[int, int]:
    idx = int(os.getenv("SHARD_INDEX", os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    cnt = int(os.getenv("SHARD_COUNT", os.getenv("SLURM_ARRAY_TASK_COUNT", "1")))
    if idx < 0 or cnt < 1 or idx >= cnt:
        raise ValueError("Invalid shard configuration")
    return idx, cnt

def _build_config() -> ExtractorConfig:
    extract_html, extract_pdf = _parse_extract_mode(os.getenv("EXTRACT_MODE"))
    shard_index, shard_count = _parse_sharding()

    return ExtractorConfig(
        input_dir=require_env_path("WARC_INPUT_DIR"),
        output_dir=require_env_path("OUTPUT_DIR"),
        seeds_xlsx=require_env_path("SEEDS_XLSX"),
        failed_warcs_file=require_env_path("FAILED_WARCS_FILE"),
        extract_html=extract_html,
        extract_pdf=extract_pdf,
        shard_index=shard_index,
        shard_count=shard_count
    )

def main() -> int:
    load_dotenv()
    logger = setup_logging()
    try:
        run_extraction(_build_config(), logger)
        return 0
    except Exception:
        logger.exception("Extraction failed")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
