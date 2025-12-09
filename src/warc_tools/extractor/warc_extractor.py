# src/extractor/warc_extractor.py
from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Set, List

from warcio.archiveiterator import ArchiveIterator

from .utils import (
    extract_text_from_html_bytes,
    extract_text_from_pdf_bytes,
    get_base_site_from_url,
    is_allowed_url,
    iter_warc_files,
    load_allowed_domains_from_xlsx,
    shard_files,
)


@dataclass
class ExtractorConfig:
    """
    Configuration for a WARC → JSONL extraction run.
    """
    input_dir: Path
    output_dir: Path
    seeds_xlsx: Path
    failed_warcs_file: Path
    shard_index: int = 0
    shard_count: int = 1


def _open_warc_stream(warc_path: Path):
    """
    Open a WARC or WARC.GZ file as a binary stream.
    """
    if warc_path.suffix == ".gz":
        return gzip.open(warc_path, "rb")
    return open(warc_path, "rb")


def process_single_warc(
    warc_path: Path,
    allowed_domains: Set[str],
    output_dir: Path,
    logger: logging.Logger,
) -> int:
    """
    Stream a single WARC (.warc or .warc.gz) and write one JSONL file
    with extracted documents.

    Returns:
        int: number of documents written.
    """
    logger.info(f"Processing WARC: {warc_path}")
    start_time = time.time()

    basename = warc_path.name

    # Derive output JSONL filename from WARC filename
    out_name = basename
    if out_name.endswith(".gz"):
        out_name = out_name[:-3]
    if out_name.endswith(".warc"):
        out_name = out_name[:-5]
    out_name = out_name + ".jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name

    num_docs = 0
    record_count = 0

    with _open_warc_stream(warc_path) as stream, out_path.open("w", encoding="utf-8") as fout:
        for record in ArchiveIterator(stream):
            record_count += 1
            try:
                # Only HTTP response records
                if record.rec_type != "response":
                    continue
                if not record.http_headers:
                    continue

                url = record.rec_headers.get_header("WARC-Target-URI")
                warc_date = record.rec_headers.get_header("WARC-Date")

                # ETHZ domain filter
                if not is_allowed_url(url, allowed_domains):
                    continue

                http_headers = record.http_headers
                status = http_headers.get_statuscode()
                if status != "200":
                    continue

                ctype = http_headers.get_header("Content-Type") or ""
                ctype_l = ctype.lower()

                is_html = "text/html" in ctype_l
                is_pdf = ("application/pdf" in ctype_l) or ("application/x-pdf" in ctype_l)

                if not (is_html or is_pdf):
                    continue

                payload = record.content_stream().read()
                if not payload:
                    continue

                if is_html:
                    text = extract_text_from_html_bytes(payload)
                else:
                    text = extract_text_from_pdf_bytes(payload, logger)

                if not text.strip():
                    continue

                base_site = get_base_site_from_url(url or "")

                doc = {
                    "url": url,
                    "capture_time": warc_date,
                    "base_site": base_site,
                    "status": status,
                    "content_type": ctype,
                    "text": text,
                    "source_warc": basename,
                }

                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                num_docs += 1

            except Exception as e:
                logger.warning(f"Record-level error in {warc_path.name}: {e}")
                continue

    elapsed = time.time() - start_time
    logger.info(
        f"Finished {warc_path.name}: {num_docs} docs written "
        f"(records seen: {record_count}, time: {elapsed:.1f}s)"
    )
    return num_docs


def run_extraction(config: ExtractorConfig, logger: logging.Logger) -> None:
    """
    Run WARC → JSONL extraction according to the given configuration.

    This is the main entry point for the library API, called by the CLI.
    """
    if not config.input_dir.exists():
        logger.error(f"WARC input directory does not exist: {config.input_dir}")
        raise FileNotFoundError(config.input_dir)
    if not config.seeds_xlsx.exists():
        logger.error(f"Seeds XLSX does not exist: {config.seeds_xlsx}")
        raise FileNotFoundError(config.seeds_xlsx)

    logger.info("=== Extraction configuration ===")
    logger.info(f"Input directory:    {config.input_dir}")
    logger.info(f"Output directory:   {config.output_dir}")
    logger.info(f"Seeds XLSX:         {config.seeds_xlsx}")
    logger.info(f"Failed WARCs file:  {config.failed_warcs_file}")
    logger.info(f"Shard index/count:  {config.shard_index}/{config.shard_count}")
    logger.info("================================")

    allowed_domains = load_allowed_domains_from_xlsx(config.seeds_xlsx, logger)

    all_warcs = sorted(iter_warc_files(config.input_dir))
    if not all_warcs:
        logger.warning(f"No .warc/.warc.gz files found in {config.input_dir}")
        return

    logger.info(f"Found {len(all_warcs)} WARC files total.")
    warcs_to_process: List[Path] = shard_files(
        all_warcs, config.shard_index, config.shard_count
    )
    logger.info(f"This shard will process {len(warcs_to_process)} WARC files.")

    failed_warcs: List[str] = []
    total_docs = 0
    start_all = time.time()

    for idx, warc_path in enumerate(warcs_to_process, start=1):
        logger.info(f"=== [{idx}/{len(warcs_to_process)}] {warc_path.name} ===")
        try:
            num_docs = process_single_warc(
                warc_path=warc_path,
                allowed_domains=allowed_domains,
                output_dir=config.output_dir,
                logger=logger,
            )
            total_docs += num_docs
        except Exception as e:
            logger.error(f"WARC-level error on {warc_path}: {e}")
            failed_warcs.append(str(warc_path))
            continue

    elapsed_all = time.time() - start_all
    logger.info(
        f"All WARCs processed for this shard. Total docs: {total_docs}, "
        f"time: {elapsed_all:.1f}s"
    )

    if failed_warcs:
        logger.warning(
            f"{len(failed_warcs)} WARCs failed in this shard. "
            f"Writing list to: {config.failed_warcs_file}"
        )
        config.failed_warcs_file.parent.mkdir(parents=True, exist_ok=True)
        # append, in case multiple shards write to the same file
        with config.failed_warcs_file.open("a", encoding="utf-8") as f:
            for w in failed_warcs:
                f.write(w + "\n")
    else:
        logger.info("No WARCs failed in this shard.")
