from __future__ import annotations

import os

from .deduplicate import DedupConfig, run_dedup
from dotenv import load_dotenv


def _env(name: str, default: str | None = None) -> str:
    v = (os.getenv(name) or "").strip()
    if v:
        return v
    if default is not None:
        return default
    raise RuntimeError(f"Missing required env var: {name}")


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
    cfg = build_config_from_env()
    run_dedup(cfg)
    return 0


def main() -> int:
    load_dotenv()
    try:
        return run_dedup_from_env()
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
