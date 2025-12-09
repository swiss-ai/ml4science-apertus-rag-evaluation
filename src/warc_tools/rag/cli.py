# src/warc_tools/rag/cli.py
from __future__ import annotations

import os
import sys
from pathlib import Path

from .utils import load_env_if_dev
from warc_tools.indexer.utils import require_env, setup_logging
from .rag_pipeline import RAGConfig, run_rag_query


def main() -> None:
    """
    CLI for running a single RAG query against ES + LLM.

    Usage:
        warc-rag "What is ETH Zurich Library?"
    or:
        python -m warc_tools.rag.cli "your question"
    or:
        export RAG_QUERY="your question"; warc-rag
    """
    load_env_if_dev()

    # Query from args or env
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = os.getenv("RAG_QUERY", "").strip()

    if not query:
        print("Usage: warc-rag \"your question\"  (or set RAG_QUERY)", file=sys.stderr)
        sys.exit(1)

    # Core config
    es_url = require_env("ES_URL")
    es_user = os.getenv("ES_USER") or None
    es_password = os.getenv("ES_PASSWORD") or None
    index_name = require_env("ES_INDEX_NAME")
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE") or None

    logger = setup_logging(log_level, log_file)

    logger.info(f"RAG query: {query!r}")
    config = RAGConfig(
        es_url=es_url,
        es_user=es_user,
        es_password=es_password,
        index_name=index_name,
        top_k=top_k
    )

    try:
        rag_result = run_rag_query(config, query, logger)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        sys.exit(1)

    # Print plain-text answer
    print("\n=== ANSWER ===\n")
    print(rag_result.answer)
    print("\n=== SOURCES (top {k}) ===".format(k=len(rag_result.nodes)))
    for i, node in enumerate(rag_result.nodes, start=1):
        meta = node.node.metadata or {}
        url = meta.get("url")
        warc = meta.get("source_warc")
        capture = meta.get("capture_time")
        print(f"[{i}] score={node.score:.4f} url={url} warc={warc} capture={capture}")


if __name__ == "__main__":
    main()
