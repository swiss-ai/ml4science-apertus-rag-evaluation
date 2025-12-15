from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

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


def _dedup_mode(input_dir: Path, conn: sqlite3.Connection) -> None:
    for fpath in _iter_jsonl_files(input_dir):
        with fpath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
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
                _upsert(conn, key, ts, line.strip())

        conn.commit()


def _dump_by_base_site_and_year(conn: sqlite3.Connection, out_root: Path) -> None:
    rows = conn.execute("SELECT json_line FROM best")
    for (line,) in rows:
        obj = json.loads(line)
        base = obj["base_site"]
        year = obj["capture_time"][:4]

        out_dir = out_root / base
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{year}.jsonl"
        with out_file.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def run_dedup(cfg: DedupConfig) -> None:
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    db_dir = Path(cfg.db_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode in {"html", "all"}:
        conn = _open_db(db_dir / "dedup_html.sqlite")
        try:
            _dedup_mode(input_root / "html", conn)
            _dump_by_base_site_and_year(conn, output_root / "html")
        finally:
            conn.close()

    if cfg.mode in {"pdf", "all"}:
        conn = _open_db(db_dir / "dedup_pdf.sqlite")
        try:
            _dedup_mode(input_root / "pdf", conn)
            _dump_by_base_site_and_year(conn, output_root / "pdf")
        finally:
            conn.close()
