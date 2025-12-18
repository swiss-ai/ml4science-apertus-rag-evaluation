"""
Extractor Utilities Module

Utility functions for WARC extraction including:
- Environment variable handling
- Logging setup
- HTML and PDF text extraction
- URL domain extraction and filtering
- WARC file iteration
"""
from __future__ import annotations

import io
import logging
import os
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

from typing import Sequence, TypeVar

warnings.filterwarnings("ignore", message=r".*Cannot set gray.*")
warnings.filterwarnings("ignore", message=r".*invalid float value.*")


def require_env_str(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

T = TypeVar("T")

def shard_items(items: Sequence[T], shard_index: int, shard_count: int) -> list[T]:
    if shard_count <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % shard_count == shard_index]

def require_env_path(name: str) -> Path:
    return Path(require_env_str(name))

def setup_logging() -> logging.Logger:
    level_name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    log_file = (os.getenv("LOG_FILE") or "").strip() or None

    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL={level_name!r}")

    logger = logging.getLogger("extractor")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(level)
            logger.addHandler(fh)

    for name in (
        "pdfminer",
        "pdfminer.psparser",
        "pdfminer.pdfinterp",
        "pdfminer.layout",
        "pdfminer.pdfcolor",
    ):
        l = logging.getLogger(name)
        l.setLevel(logging.ERROR)
        l.propagate = False

    return logger

def get_base_site_from_url(url: str) -> str:
    if "//" in url:
        url = url.split("//", 1)[1]
    for p in ("www.", "www0.", "www1.", "www2.", "www3."):
        if url.startswith(p):
            url = url[len(p):]
    return url.split("/", 1)[0].split(":", 1)[0].rstrip(".")

def load_allowed_domains_from_xlsx(path: Path, logger: logging.Logger) -> set[str]:
    df = pd.read_excel(path).fillna("")
    urls = [u for u in df.get("URL", []) if isinstance(u, str) and u.strip()]
    domains = set()
    for u in urls:
        try:
            domains.add(get_base_site_from_url(u).lower())
        except Exception:
            continue
    logger.info("Loaded %d allowed domains", len(domains))
    return domains

def is_allowed_url(url: str | None, allowed: set[str]) -> bool:
    return bool(url) and get_base_site_from_url(url).lower() in allowed

def extract_text_from_html_bytes(payload: bytes) -> str:
    html = payload.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.extract()
    text = soup.get_text()
    lines = [l.strip() for l in text.splitlines()]
    chunks = [p.strip() for l in lines for p in l.split("  ")]
    out = "\n".join(c for c in chunks if c).strip()
    return "" if out in {"", "Redirecting"} else out

def extract_text_from_pdf_bytes(
    payload: bytes,
    logger: logging.Logger,
    *,
    max_pages: int = 50,
    max_bytes: int = 25_000_000,
) -> str:
    if len(payload) > max_bytes:
        return ""
    try:
        chunks: list[str] = []
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            if getattr(pdf, "is_encrypted", False):
                return ""
            for page in pdf.pages[:max_pages]:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    continue
                if t.strip():
                    chunks.append(t)
        return "\n".join(chunks).strip()
    except Exception as e:
        logger.debug("PDF extraction error: %s", e)
        return ""

def iter_warc_files(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*"):
        if p.is_file() and (p.name.endswith(".warc") or p.name.endswith(".warc.gz")):
            yield p
