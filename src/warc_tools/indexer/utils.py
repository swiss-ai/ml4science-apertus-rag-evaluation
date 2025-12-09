# src/warc_tools/indexer/utils.py
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, List, Dict, Any

from elasticsearch import Elasticsearch

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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

    # Remove any existing handlers to avoid duplicates
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


def get_embedding_model_from_env(logger: logging.Logger):
    """
    Choose an embedding backend based on env vars:

    EMBED_PROVIDER:
      - "ollama" : local Ollama server
      - "cscs"   : OpenAI-compatible endpoint (e.g. SwissAI)
      - "openai" : OpenAI-compatible endpoint
      - "hf"     : local / HF Hub model via HuggingFaceEmbedding

    EMBED_MODEL:      model name / id
    EMBED_BASE_URL:   (for cscs/openai) base URL of the OpenAI-compatible API
    EMBED_API_KEY:    (for cscs/openai) API key/token
    EMBED_TIMEOUT:    optional seconds (default 60)
    """
    provider = require_env("EMBED_PROVIDER").lower()
    model = require_env("EMBED_MODEL")
    timeout = int(os.getenv("EMBED_TIMEOUT", "60"))

    logger.info(f"Using embedding provider='{provider}', model='{model}'")

    if provider == "ollama":
        return OllamaEmbedding(model_name=model, request_timeout=timeout)

    elif provider in ("openai", "cscs"):
        base_url = require_env("EMBED_BASE_URL")
        api_key = require_env("EMBED_API_KEY")

        logger.info(f"Using OpenAI-compatible embeddings at {base_url}")
        return OpenAIEmbedding(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider in ("hf", "huggingface"):
        return HuggingFaceEmbedding(model_name=model)

    else:
        logger.error(f"Unsupported EMBED_PROVIDER: {provider}")
        sys.exit(1)


def batched(iterable, batch_size: int):
    """
    Yield lists of size <= batch_size from iterable.
    """
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_es_index_for_vectors(
    es_client: Elasticsearch,
    index_name: str,
    dim: int,
    logger: logging.Logger,
) -> None:
    """
    Create an ES index suitable for dense vectors + metadata.
    """
    if es_client.indices.exists(index=index_name):
        logger.info(f"Index '{index_name}' already exists, deleting it...")
        es_client.indices.delete(index=index_name)

    logger.info(f"Creating vector index '{index_name}' with dim={dim}...")
    body: Dict[str, Any] = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "url": {"type": "keyword"},
                "capture_time": {"type": "date", "ignore_malformed": True},
                "base_site": {"type": "keyword"},
                "status": {"type": "keyword"},
                "content_type": {"type": "keyword"},
                "source_warc": {"type": "keyword"},
                # vector field is managed by ElasticsearchStore
            }
        },
    }
    es_client.indices.create(index=index_name, body=body)
