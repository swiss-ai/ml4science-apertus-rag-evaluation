# src/warc_tools/rag/rag_pipeline.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from warc_tools.indexer.utils import get_embedding_model_from_env
from .utils import get_llm_from_env


@dataclass
class RAGConfig:
    es_url: str
    es_user: str
    es_password: str
    index_name: str
    top_k: int = 5


@dataclass
class RAGResult:
    answer: str
    nodes: List[NodeWithScore]

def _build_es_store(config: "RAGConfig", logger: logging.Logger) -> ElasticsearchStore:
    """
    Build an ElasticsearchStore with optional basic auth and TLS settings.
    Uses the same env style as the indexer:
      ES_USER, ES_PASSWORD, ES_VERIFY_CERTS
    """
    es_user = os.getenv("ES_USER") or None
    es_password = os.getenv("ES_PASSWORD") or None
    es_verify_certs = os.getenv("ES_VERIFY_CERTS", "false").lower() in ("1", "true", "yes")

    kwargs = {
        "es_url": config.es_url,
        "index_name": config.index_name,
    }

    if es_user and es_password:
        logger.info("Using basic auth for Elasticsearch in RAG.")
        kwargs["es_user"] = es_user
        kwargs["es_password"] = es_password
        kwargs["verify_certs"] = es_verify_certs
    else:
        if es_verify_certs:
            kwargs["verify_certs"] = es_verify_certs

    return ElasticsearchStore(**kwargs)


def build_index_for_rag(config: RAGConfig, logger: logging.Logger) -> VectorStoreIndex:
    """
    Build a LlamaIndex VectorStoreIndex over the existing Elasticsearch index.
    Assumes embeddings were already stored there with the same embed model.
    """
    logger.info(f"Initializing ElasticsearchStore for RAG (index={config.index_name})")

    vector_store = _build_es_store(config, logger)

    embed_model = get_embedding_model_from_env(logger)
    llm = get_llm_from_env(logger)

    # Build index using the existing vector store; no re-embedding.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Attach LLM + return index (we’ll create a query engine in run_rag)
    index._llm = llm  # optional; we'll still pass llm explicitly
    return index


def run_rag_query(
    config: RAGConfig,
    query: str,
    logger: logging.Logger,
) -> RAGResult:
    """
    Run a single RAG query against Elasticsearch-backed vector store + LLM.
    """
    logger.info(f"Running RAG query (top_k={config.top_k}): {query!r}")

    vector_store = _build_es_store(config, logger)
    
    embed_model = get_embedding_model_from_env(logger)
    llm = get_llm_from_env(logger)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    query_engine = index.as_query_engine(
        similarity_top_k=config.top_k,
        llm=llm,
    )

    response = query_engine.query(query)

    # response.source_nodes is a list[NodeWithScore]
    nodes: List[NodeWithScore] = list(response.source_nodes)

    logger.info(f"RAG answer length: {len(str(response))} chars")
    logger.info(f"Retrieved {len(nodes)} source chunks.")

    return RAGResult(
        answer=str(response),
        nodes=nodes,
    )
