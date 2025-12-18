"""
Extractor Configuration Module

Defines the configuration dataclass for WARC extraction process.
Contains settings for input/output directories, extraction modes, and sharding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExtractorConfig:
    input_dir: Path
    output_dir: Path
    seeds_xlsx: Path
    failed_warcs_file: Path
    extract_html: bool
    extract_pdf: bool
    shard_index: int
    shard_count: int

    @property
    def html_output_dir(self) -> Optional[Path]:
        return self.output_dir / "html" if self.extract_html else None

    @property
    def pdf_output_dir(self) -> Optional[Path]:
        return self.output_dir / "pdf" if self.extract_pdf else None
