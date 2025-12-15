from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from .deduplicate import DedupConfig, run_dedup


def _env(name: str, default: str | None = None) -> str:
    v = (os.getenv(name) or "").strip()
    if v:
        return v
    if default is not None:
        return default
    raise RuntimeError(f"Missing required env var: {name}")


def _setup_logging() -> logging.Logger:
    level = _env("LOG_LEVEL", "INFO").upper()
    lvl = getattr(logging, level, logging.INFO)

    logger = logging.getLogger("deduper")
    logger.setLevel(lvl)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(lvl)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    return logger


def build_config_from_env() -> DedupConfig:
    mode = _env("DEDUP_MODE", "all").lower()
    if mode not in {"html", "pdf", "all"}:
        raise RuntimeError("DEDUP_MODE must be one of: html, pdf, all")

    output_root = _env("DEDUP_OUTPUT_ROOT")
    return DedupConfig(
        input_root=_env("DEDUP_INPUT_ROOT"),
        output_root=output_root,
        mode=mode,
        db_dir=_env("DEDUP_DB_DIR", os.path.join(output_root, ".dedup_db")),
    )


def run_dedup_from_env() -> int:
    logger = _setup_logging()
    cfg = build_config_from_env()
    logger.info("Starting dedup: mode=%s input=%s output=%s", cfg.mode, cfg.input_root, cfg.output_root)
    run_dedup(cfg, logger)
    logger.info("Dedup finished")
    return 0


def main() -> int:
    load_dotenv()
    try:
        return run_dedup_from_env()
    except Exception:
        logging.getLogger("deduper").exception("Dedup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
