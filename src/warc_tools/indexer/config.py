from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexerConfig:
    jsonl_dir: Path
    es_url: str
    index_name: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    failed_warcs_file: Path
    es_user: str | None = None
    es_password: str | None = None
    es_verify_certs: bool = False
    no_recreate_index: bool = False
