from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from collections import defaultdict

@dataclass(frozen=True)
class DedupConfig:
    input_root: str
    output_root: str
    mode: str
    db_dir: str


def _iter_jsonl_files(dir_path: Path) -> Iterable[Path]:
    if dir_path.exists():
        yield from sorted(p for p in dir_path.rglob("*.jsonl") if p.is_file())


def _parse_capture_time(s: str | None) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS best (
            k TEXT PRIMARY KEY,
            capture_ts INTEGER NOT NULL,
            json_line TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _upsert(conn: sqlite3.Connection, key: str, ts: int, line: str) -> None:
    row = conn.execute("SELECT capture_ts FROM best WHERE k = ?", (key,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO best (k, capture_ts, json_line) VALUES (?, ?, ?)",
            (key, ts, line),
        )
    elif ts > int(row[0]):
        conn.execute(
            "UPDATE best SET capture_ts = ?, json_line = ? WHERE k = ?",
            (ts, line, key),
        )


def _dedup_mode(input_dir: Path, conn: sqlite3.Connection, logger: logging.Logger, label: str) -> None:
    for fpath in _iter_jsonl_files(input_dir):
        logger.info("[%s] START %s", label, fpath.name)
        seen = kept = 0
        try:
            with fpath.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    seen += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    url = obj.get("url")
                    base = obj.get("base_site")
                    dt = _parse_capture_time(obj.get("capture_time"))
                    if not url or not base or dt is None:
                        continue

                    key = f"{url}||{dt.year}"
                    ts = int(dt.timestamp())
                    _upsert(conn, key, ts, line)
                    kept += 1

            conn.commit()
            logger.info("[%s] DONE  %s seen=%d kept=%d", label, fpath.name, seen, kept)
        except Exception as e:
            logger.error("[%s] ERROR %s: %s (continuing)", label, fpath, e)
            try:
                conn.commit()
            except Exception:
                pass
            continue


def _clear_output_dir(out_root: Path, logger: logging.Logger, label: str) -> None:
    if out_root.exists():
        logger.info("[%s] Clearing output dir: %s", label, out_root)
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def _dump_by_base_site_and_year(conn: sqlite3.Connection, out_root: Path, logger, label: str) -> None:
    _clear_output_dir(out_root, logger, label)

    rows = conn.execute("SELECT json_line FROM best")

    buffers = defaultdict(list)   # (base, year) -> [lines...]
    counts = defaultdict(int)     # (base, year) -> total written
    written = 0
    skipped = 0

    FLUSH_LINES = 2000

    def flush(key: tuple[str, str]) -> None:
        base, year = key
        lines = buffers.get(key)
        if not lines:
            return

        out_dir = out_root / base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{year}.jsonl"

        with out_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

        counts[key] += len(lines)
        buffers[key].clear()

    for (line,) in rows:
        try:
            obj = json.loads(line)
            base = obj["base_site"]
            year = obj["capture_time"][:4]

            key = (base, year)
            buffers[key].append(line)
            written += 1

            if len(buffers[key]) >= FLUSH_LINES:
                flush(key)

        except Exception:
            skipped += 1
            continue

    # flush remaining buffers
    for key in list(buffers.keys()):
        flush(key)

    for (base, year), n in sorted(counts.items()):
        logger.info("[%s] WRITE %s %s records=%d → %s", label, base, year, n, out_root / base / f"{year}.jsonl")

    logger.info("[%s] WRITE SUMMARY unique_keys=%d files=%d skipped=%d", label, written, len(counts), skipped)

def run_dedup(cfg: DedupConfig, logger: logging.Logger) -> None:
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    db_dir = Path(cfg.db_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode in {"html", "all"}:
        conn = _open_db(db_dir / "dedup_html.sqlite")
        try:
            _dedup_mode(input_root / "html", conn, logger, "html")
            _dump_by_base_site_and_year(conn, output_root / "html", logger, "html")
        finally:
            conn.close()

    if cfg.mode in {"pdf", "all"}:
        conn = _open_db(db_dir / "dedup_pdf.sqlite")
        try:
            _dedup_mode(input_root / "pdf", conn, logger, "pdf")
            _dump_by_base_site_and_year(conn, output_root / "pdf", logger, "pdf")
        finally:
            conn.close()
