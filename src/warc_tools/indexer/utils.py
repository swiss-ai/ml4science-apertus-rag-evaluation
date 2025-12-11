# src/warc_tools/indexer/utils.py
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, List, Dict, Any

import openai

#from openai import OpenAI # CSCS / OpenAI-compatible client

from elasticsearch import Elasticsearch

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class CSCSOpenAICompatibleEmbedding(BaseEmbedding):
    """
    Embedding wrapper for an OpenAI-compatible /embeddings endpoint hosted at CSCS.

    Uses the official `openai` client with a custom base_url and arbitrary model name.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> None:
        super().__init__()
        self._model = model
        self._client = openai.Client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    # ---- sync methods ----
    def _get_query_embedding(self, query: str) -> Embedding:
        resp = self._client.embeddings.create(
            model=self._model,
            input=query,
        )
        return resp.data[0].embedding

    def _get_text_embedding(self, text: str) -> Embedding:
        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return resp.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        # CSCS/OpenAI API supports batching
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # ---- async methods ----
    async def _aget_query_embedding(self, query: str) -> Embedding:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=query,
        )
        return resp.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> Embedding:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return resp.data[0].embedding

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

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
      - "cscs"   : OpenAI-compatible endpoint (e.g. SwissAI) for embeddings
      - "openai" : OpenAI embeddings
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

    elif provider == "cscs":
        base_url = require_env("EMBED_BASE_URL")
        api_key = require_env("EMBED_API_KEY")

        logger.info(f"Using CSCS OpenAI-compatible embeddings at {base_url}")
        return CSCSOpenAICompatibleEmbedding(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider == "openai":
        # Real OpenAI; base_url can be omitted or overridden via env
        api_key = require_env("EMBED_API_KEY")
        base_url = os.getenv("EMBED_BASE_URL")  # usually None for OpenAI

        logger.info(f"Using OpenAI embeddings (base_url={base_url or 'default'})")
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
