"""
WARC Extractor Module

Main extraction logic for processing WARC files and extracting HTML/PDF content.
Handles WARC file parsing, content extraction, and JSONL output generation.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Optional

from warcio.archiveiterator import ArchiveIterator

from .config import ExtractorConfig
from .utils import (
    extract_text_from_html_bytes,
    extract_text_from_pdf_bytes,
    get_base_site_from_url,
    is_allowed_url,
    iter_warc_files,
    load_allowed_domains_from_xlsx,
    shard_items
)


def _open_warc(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _out_name(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".warc"):
        name = name[:-5]
    return name + ".jsonl"


def _content_kind(ctype: str) -> Optional[str]:
    c = ctype.lower()
    if "text/html" in c:
        return "html"
    if "application/pdf" in c or "application/x-pdf" in c:
        return "pdf"
    return None


def _write_jsonl(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)

def _warc_already_done(
    warc_path: Path,
    *,
    html_dir: Optional[Path],
    pdf_dir: Optional[Path],
    extract_html: bool,
    extract_pdf: bool,
) -> bool:
    out = _out_name(warc_path)
    html_path = (html_dir / out) if extract_html and html_dir else None
    pdf_path = (pdf_dir / out) if extract_pdf and pdf_dir else None
    return bool((html_path and html_path.exists()) or (pdf_path and pdf_path.exists()))

def process_single_warc(
    warc_path: Path,
    *,
    allowed_domains: set[str],
    html_dir: Optional[Path],
    pdf_dir: Optional[Path],
    extract_html: bool,
    extract_pdf: bool,
    logger: logging.Logger,
) -> tuple[int, int]:
    start = time.time()

    html_docs = 0
    pdf_docs = 0

    html_path = html_dir / _out_name(warc_path) if extract_html and html_dir else None
    pdf_path = pdf_dir / _out_name(warc_path) if extract_pdf and pdf_dir else None

    try:
        with _open_warc(warc_path) as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type != "response" or not record.http_headers:
                    continue

                url = record.rec_headers.get_header("WARC-Target-URI")
                if not is_allowed_url(url, allowed_domains):
                    continue

                if record.http_headers.get_statuscode() != "200":
                    continue

                ctype = record.http_headers.get_header("Content-Type") or ""
                kind = _content_kind(ctype)
                if kind == "html" and not html_path:
                    continue
                if kind == "pdf" and not pdf_path:
                    continue
                if kind is None:
                    continue

                payload = record.content_stream().read()
                if not payload:
                    continue

                text = (
                    extract_text_from_html_bytes(payload)
                    if kind == "html"
                    else extract_text_from_pdf_bytes(payload, logger)
                )
                if not text.strip():
                    continue

                doc = {
                    "url": url,
                    "capture_time": record.rec_headers.get_header("WARC-Date"),
                    "base_site": get_base_site_from_url(url or ""),
                    "status": 200,
                    "content_type": ctype,
                    "text": text,
                    "source_warc": warc_path.name,
                }
                line = json.dumps(doc, ensure_ascii=False) + "\n"

                if kind == "html":
                    _write_jsonl(html_path, line)
                    html_docs += 1
                else:
                    _write_jsonl(pdf_path, line)
                    pdf_docs += 1

        return html_docs, pdf_docs

    except Exception:
        if html_path and html_path.exists():
            html_path.unlink()
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()
        raise

    finally:
        logger.info(
            "Finished %s html=%d pdf=%d time=%.1fs",
            warc_path.name,
            html_docs,
            pdf_docs,
            time.time() - start,
        )


def run_extraction(config: ExtractorConfig, logger: logging.Logger) -> None:
    allowed = load_allowed_domains_from_xlsx(config.seeds_xlsx, logger)

    all_warcs = sorted(iter_warc_files(config.input_dir))
    warcs = shard_items(all_warcs, config.shard_index, config.shard_count)

    logger.info(
        "WARCs total=%d shard=%d/%d processing=%d",
        len(all_warcs),
        config.shard_index,
        config.shard_count,
        len(warcs),
    )

    failed: list[str] = []

    for i, warc in enumerate(warcs, 1):
        if _warc_already_done(
            warc,
            html_dir=config.html_output_dir,
            pdf_dir=config.pdf_output_dir,
            extract_html=config.extract_html,
            extract_pdf=config.extract_pdf,
        ):
            logger.info("[%d/%d] %s (skipped: output exists)", i, len(warcs), warc.name)
            continue
        
        logger.info("[%d/%d] %s", i, len(warcs), warc.name)
        try:
            process_single_warc(
                warc,
                allowed_domains=allowed,
                html_dir=config.html_output_dir,
                pdf_dir=config.pdf_output_dir,
                extract_html=config.extract_html,
                extract_pdf=config.extract_pdf,
                logger=logger,
            )
        except Exception as e:
            logger.error("Failed %s: %s", warc, e)
            failed.append(str(warc))

    if failed:
        config.failed_warcs_file.parent.mkdir(parents=True, exist_ok=True)
        with config.failed_warcs_file.open("a", encoding="utf-8") as f:
            for w in failed:
                f.write(w + "\n")
