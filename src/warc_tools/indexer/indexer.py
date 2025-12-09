# src/warc_tools/indexer/indexer.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from elasticsearch import Elasticsearch

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from .utils import (
    batched,
    create_es_index_for_vectors,
    get_embedding_model_from_env,
)


@dataclass
class IndexerConfig:
    """
    Configuration for JSONL → Elasticsearch vector indexing.
    """
    jsonl_dir: Path
    es_url: str
    index_name: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    failed_warcs_file: Path

    # Optional ES auth / TLS
    es_user: str | None = None
    es_password: str | None = None
    es_verify_certs: bool = False

    # Index recreation
    no_recreate_index: bool = False


def iter_documents_from_jsonl(
    jsonl_dir: Path,
    logger: logging.Logger,
):
    """
    Stream LlamaIndex Documents from all *.jsonl files in jsonl_dir.
    One Document per JSON object (line).
    """
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files in {jsonl_dir}")

    for file_idx, jsonl_path in enumerate(jsonl_files, start=1):
        logger.info(f"[FILE {file_idx}/{len(jsonl_files)}] {jsonl_path.name}")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        f"JSON decode error in {jsonl_path.name} line {line_idx}, skipping"
                    )
                    continue

                text = (obj.get("text") or "").strip()
                if not text:
                    continue

                metadata = {
                    "url": obj.get("url"),
                    "capture_time": obj.get("capture_time"),
                    "base_site": obj.get("base_site"),
                    "status": obj.get("status"),
                    "content_type": obj.get("content_type"),
                    "source_warc": obj.get("source_warc"),
                    "jsonl_file": str(jsonl_path),
                }

                yield Document(text=text, metadata=metadata)


def run_indexing(config: IndexerConfig, logger: logging.Logger) -> None:
    """
    Run JSONL → Elasticsearch vector indexing according to the given configuration.
    """
    if not config.jsonl_dir.exists():
        logger.error(f"JSONL directory does not exist: {config.jsonl_dir}")
        raise FileNotFoundError(config.jsonl_dir)

    logger.info("=== Indexer configuration ===")
    logger.info(f"JSONL_DIR:                 {config.jsonl_dir}")
    logger.info(f"ES_URL:                    {config.es_url}")
    logger.info(f"ES_INDEX_NAME:             {config.index_name}")
    logger.info(f"BATCH_SIZE:                {config.batch_size}")
    logger.info(f"CHUNK_SIZE:                {config.chunk_size}")
    logger.info(f"CHUNK_OVERLAP:             {config.chunk_overlap}")
    logger.info(f"FAILED_WARCS_INDEXING_FILE:{config.failed_warcs_file}")
    logger.info(f"NO_RECREATE_INDEX:         {config.no_recreate_index}")
    logger.info("============================")

    # --- Elasticsearch connection ---
    if config.es_user and config.es_password:
        es_client = Elasticsearch(
            config.es_url,
            basic_auth=(config.es_user, config.es_password),
            verify_certs=config.es_verify_certs,
        )
    else:
        es_client = Elasticsearch(config.es_url, verify_certs=config.es_verify_certs)

    logger.info("Pinging Elasticsearch...")
    if not es_client.ping():
        logger.error(f"Could not connect to Elasticsearch at {config.es_url}")
        raise RuntimeError("Elasticsearch ping failed")
    logger.info("Elasticsearch is reachable.")

    # --- Embedding model from env ---
    embed_model = get_embedding_model_from_env(logger)

    # Determine embedding dimension
    try:
        logger.info("Testing embedding call to determine dimension...")
        test_vec = embed_model.get_text_embedding("hello world")
        dim = len(test_vec)
        logger.info(f"Embedding backend OK, dimension = {dim}")
    except Exception as e:
        logger.error(f"Error calling embedding model: {e}")
        raise

    # --- Create / reuse ES index for vectors ---
    if config.no_recreate_index:
        logger.info("NO_RECREATE_INDEX=1 -> skipping index recreation.")
        if not es_client.indices.exists(index=config.index_name):
            logger.info(f"Index '{config.index_name}' does not exist, creating...")
            create_es_index_for_vectors(es_client, config.index_name, dim, logger)
    else:
        create_es_index_for_vectors(es_client, config.index_name, dim, logger)

    # --- Vector store ---
    es_store_kwargs = {
        "es_url": config.es_url,
        "index_name": config.index_name,
    }
    if config.es_user and config.es_password:
        es_store_kwargs["es_user"] = config.es_user
        es_store_kwargs["es_password"] = config.es_password
        es_store_kwargs["verify_certs"] = config.es_verify_certs

    logger.info("Initializing ElasticsearchStore...")
    vector_store = ElasticsearchStore(**es_store_kwargs)

    # --- Chunker + pipeline ---
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        vector_store=vector_store,
    )

    # --- Indexing loop with tracking failed WARCs ---
    docs_iter = iter_documents_from_jsonl(config.jsonl_dir, logger)
    total_docs = 0          # number of original documents (pre-chunk)
    batch_idx = 0
    start_all = time.time()
    failed_warcs: Set[str] = set()

    for batch in batched(docs_iter, config.batch_size):
        batch_idx += 1
        logger.info(
            f"[BATCH {batch_idx}] Processing {len(batch)} full documents "
            f"(seen so far: {total_docs})..."
        )
        t0 = time.time()
        try:
            pipeline.run(documents=batch)
        except Exception as e:
            logger.error(f"Error in pipeline.run for batch {batch_idx}: {e}")
            # collect source_warc values for this failed batch
            for doc in batch:
                sw = (doc.metadata or {}).get("source_warc")
                if sw:
                    failed_warcs.add(sw)
            continue

        dt = time.time() - t0
        total_docs += len(batch)
        logger.info(
            f"[BATCH {batch_idx}] Done. Batch time: {dt:.1f}s, "
            f"total docs ingested (pre-chunk): {total_docs}"
        )

    total_time = time.time() - start_all
    logger.info(
        f"Finished indexing. Total docs (pre-chunk) processed: {total_docs}, "
        f"total time: {total_time:.1f}s"
    )

    # --- Write failed WARCs (indexing failures) to file ---
    if failed_warcs:
        logger.warning(
            f"{len(failed_warcs)} WARCs had batches that failed during indexing. "
            f"Writing list to: {config.failed_warcs_file}"
        )
        config.failed_warcs_file.parent.mkdir(parents=True, exist_ok=True)
        with config.failed_warcs_file.open("w", encoding="utf-8") as f:
            for w in sorted(failed_warcs):
                f.write(w + "\n")
    else:
        logger.info("No WARCs failed during indexing.")
