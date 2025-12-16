from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Iterator, List, Sequence, TypeVar

import openai
from elasticsearch import Elasticsearch
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import BaseEmbedding

T = TypeVar("T")


class CSCSOpenAICompatibleEmbedding(BaseEmbedding):
    """
    Embedding wrapper for an OpenAI-compatible /embeddings endpoint hosted at CSCS.
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

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_embedding_with_retry(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_embedding_with_retry(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings_with_retry(texts)

    def _get_embedding_with_retry(self, text: str, max_retries: int = 5, retry_delay: float = 2.0) -> list[float]:
        """Get embedding with retry logic for 500 errors."""
        import time
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.embeddings.create(model=self._model, input=text)
                return resp.data[0].embedding
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if it's a 500 error (server issue, worth retrying)
                if "500" in error_str or "Internal Server Error" in error_str:
                    if attempt < max_retries:
                        wait_time = retry_delay * attempt  # Exponential backoff
                        logging.warning(
                            f"Snowflake embedding API 500 error (attempt {attempt}/{max_retries}). "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                # For other errors or max retries reached, raise
                raise
        
        # If we exhausted retries, raise the last error
        if last_error:
            raise RuntimeError(
                f"Snowflake embedding API failed after {max_retries} retries. "
                f"Last error: {last_error}. "
                f"The Snowflake model may not be running on the cluster."
            ) from last_error
        return []

    def _get_embeddings_with_retry(self, texts: list[str], max_retries: int = 5, retry_delay: float = 2.0) -> list[list[float]]:
        """Get batch embeddings with retry logic."""
        import time
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in resp.data]
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if it's a 500 error (server issue, worth retrying)
                if "500" in error_str or "Internal Server Error" in error_str:
                    if attempt < max_retries:
                        wait_time = retry_delay * attempt  # Exponential backoff
                        logging.warning(
                            f"Snowflake embedding API 500 error (attempt {attempt}/{max_retries}). "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                # For other errors or max retries reached, raise
                raise
        
        # If we exhausted retries, raise the last error
        if last_error:
            raise RuntimeError(
                f"Snowflake embedding API failed after {max_retries} retries. "
                f"Last error: {last_error}. "
                f"The Snowflake model may not be running on the cluster."
            ) from last_error
        return []

    async def _aget_query_embedding(self, query: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=query)
        return resp.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


def require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(lvl)

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
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(lvl)
        logger.addHandler(fh)

    return logger


def get_embedding_model_from_env(logger: logging.Logger) -> BaseEmbedding:
    """
    Required env vars:
      EMBED_MODEL
      EMBED_API_KEY
      EMBED_BASE_URL
    """
    model = require_env("EMBED_MODEL")
    api_key = require_env("EMBED_API_KEY")
    base_url = require_env("EMBED_BASE_URL")
    timeout = int(os.getenv("EMBED_TIMEOUT", "60"))

    logger.info("Using CSCS OpenAI-compatible embeddings model='%s'", model)

    return CSCSOpenAICompatibleEmbedding(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def batched(items: Sequence[T] | Iterator[T], batch_size: int) -> Iterator[List[T]]:
    batch: List[T] = []
    for item in items:
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
    recreate: bool = True,
) -> None:
    if recreate and es_client.indices.exists(index=index_name):
        logger.info("Index '%s' exists; deleting", index_name)
        es_client.indices.delete(index=index_name)

    if not recreate and es_client.indices.exists(index=index_name):
        logger.info("Index '%s' exists; keeping (recreate disabled)", index_name)
        return

    logger.info("Creating vector index '%s' dim=%d", index_name, dim)

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
                "modality": {"type": "keyword"},  # html | pdf
                "year": {"type": "integer"},      # derived at index time
            }
        },
    }

    es_client.indices.create(index=index_name, body=body)
