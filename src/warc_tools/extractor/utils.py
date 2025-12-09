# src/extractor/utils.py
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Set
import pandas as pd
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator

import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)
logging.getLogger("pdfminer.layout").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfcolor").setLevel(logging.ERROR)

import pdfplumber

def require_env(name: str) -> str:
    """
    Get required environment variable or exit with a clear error.
    """
    value = os.getenv(name)
    if not value:
        print(f"[ERROR] Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Configure logging to stdout and optionally to a file.
    Returns the root logger.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(lvl)

    # Remove any existing handlers to avoid duplicates if called twice
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(lvl)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(lvl)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------
# URL / domain helpers
# --------------------------------------------------------------------


def get_base_site_from_url(url_in: str) -> str:
    """
    Extract the base site from a URL.

    Example:
        "http://ethz.ch/about/test.png" -> "ethz.ch"

    This mirrors the logic you used in prep_warc_files.py.
    """
    if "//" not in url_in:
        base_site = url_in
    else:
        url_in_old = url_in
        base_site = url_in.split("//", 1)[1]
        if base_site == "":
            # e.g. urls like: "https://ethz.ch/%0ahttps://ethz.ch/"
            parts = url_in_old.split("//")
            for s in parts:
                if s:
                    base_site = s
                    break

    # strip common www prefixes
    for prefix in ("www.", "www0.", "www1.", "www2.", "www3."):
        if base_site.startswith(prefix):
            base_site = base_site[len(prefix) :]

    # remove port and path
    base_site = base_site.split(":", 1)[0]
    base_site = base_site.split("/", 1)[0]

    if base_site.endswith("."):
        base_site = base_site[:-1]

    return base_site


def load_allowed_domains_from_xlsx(xlsx_path: Path, logger: logging.Logger) -> Set[str]:
    """
    Load ETHZ seed URLs from Excel and build a set of allowed base sites (domains).
    """
    logger.info(f"Loading seed URLs from: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    df = df.fillna("")
    urls = [u for u in df["URL"] if isinstance(u, str) and u.strip()]

    domains: Set[str] = set()
    for url in urls:
        try:
            base = get_base_site_from_url(url)
            if base:
                domains.add(base.lower())
        except Exception:
            continue

    logger.info(f"Loaded {len(domains)} allowed domains from seeds.")
    return domains


def is_allowed_url(url: str | None, allowed_domains: Set[str]) -> bool:
    """
    Check whether URL belongs to one of the ETHZ seeds (by host).
    """
    if not url:
        return False
    host = get_base_site_from_url(url).lower()
    return host in allowed_domains


# --------------------------------------------------------------------
# Text extraction helpers
# --------------------------------------------------------------------


def extract_text_from_html_bytes(payload: bytes) -> str:
    """
    Extract text from raw HTML bytes using BeautifulSoup.
    """
    if not payload:
        return ""

    try:
        html = payload.decode("utf-8", errors="replace")
    except Exception:
        html = payload.decode("latin-1", errors="replace")

    if not html.strip():
        return ""

    soup = BeautifulSoup(html, features="html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text()

    # Normalize whitespace: strip lines and collapse multiple spaces
    lines = [line.strip() for line in text.splitlines()]
    chunks: List[str] = [
        phrase.strip()
        for line in lines
        for phrase in line.split("  ")
    ]
    text = "\n".join(chunk for chunk in chunks if chunk)

    if text in ("", "Redirecting"):
        return ""

    return text


def extract_text_from_pdf_bytes(payload: bytes, logger: logging.Logger) -> str:
    """
    Extract text from PDF bytes using pdfplumber.

    Returns an empty string on failure.
    """
    if not payload:
        return ""

    try:
        text_chunks: List[str] = []
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Skipping bad PDF page: {e}")
                    continue
                if page_text.strip():
                    text_chunks.append(page_text)
        return "\n".join(text_chunks).strip()
    except Exception as e:
        logger.warning(f"PDF extraction error: {e}")
        return ""


# --------------------------------------------------------------------
# WARC file iteration / sharding
# --------------------------------------------------------------------


def iter_warc_files(input_dir: Path) -> Iterable[Path]:
    """
    Yield all .warc / .warc.gz files recursively from input_dir.
    """
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".warc") or fname.endswith(".warc.gz"):
                yield Path(root) / fname


def shard_files(
    files: Sequence[Path],
    shard_index: int,
    shard_count: int,
) -> List[Path]:
    """
    Given a sequence of files, return only those that belong to this shard.

    Sharding scheme:
        file i belongs to shard (i % shard_count)
    """
    if shard_count <= 1:
        return list(files)

    selected: List[Path] = []
    for i, f in enumerate(files):
        if i % shard_count == shard_index:
            selected.append(f)
    return selected
