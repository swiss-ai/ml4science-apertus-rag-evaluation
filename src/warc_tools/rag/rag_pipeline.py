# src/warc_tools/rag/rag_pipeline.py
from __future__ import annotations

import os
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from warc_tools.indexer.utils import get_embedding_model_from_env
from .utils import get_llm_from_env

# Suppress harmless metadata parsing warnings from ElasticsearchStore
# These occur because our metadata format doesn't include _node_content/_node_type,
# but the system correctly falls back to legacy format and still works
# We suppress at logger level since these are logged warnings, not Python warnings
_es_utils_logger = logging.getLogger("llama_index.vector_stores.elasticsearch.utils")
_es_utils_logger.setLevel(logging.ERROR)  # Only show ERROR and above, suppress WARNING


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
        "text_field": "text",  # Match the field name used by the indexer
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

    try:
        response = query_engine.query(query)

        # response.source_nodes is a list[NodeWithScore]
        nodes: List[NodeWithScore] = list(response.source_nodes)

        # Extract the actual response text (handle both Response and other types)
        if hasattr(response, 'response'):
            answer_text = response.response or ""
        else:
            answer_text = str(response)
        
        # Clean up any obvious duplications (simple heuristic)
        # Split by double newlines and keep unique paragraphs
        if answer_text:
            paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
            # Remove duplicates and very similar paragraphs
            unique_paragraphs = []
            seen_starts = set()  # Track paragraph starts to detect duplicates
            
            for para in paragraphs:
                # Skip empty or very short paragraphs
                if len(para) < 20:
                    continue
                
                # Get first 15 words as a signature
                words = para.split()[:15]
                para_start = ' '.join(words).lower()
                
                # Skip if we've seen a very similar paragraph start
                is_duplicate = False
                for seen_start in seen_starts:
                    # Check if they share many common words at the start
                    seen_words = set(seen_start.split())
                    curr_words = set(para_start.split())
                    overlap = len(seen_words & curr_words)
                    if overlap >= 8 and len(para) > 50:  # Substantial overlap
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_paragraphs.append(para)
                    seen_starts.add(para_start)
            
            # If we still have multiple similar paragraphs, just take the first substantial one
            if len(unique_paragraphs) > 1:
                # Find the first paragraph that's not a note/disclaimer
                main_answer = None
                for para in unique_paragraphs:
                    para_lower = para.lower()
                    if not any(skip in para_lower for skip in ['note:', 'disclaimer', 'based on the provided', 'may not reflect']):
                        main_answer = para
                        break
                
                if main_answer:
                    answer_text = main_answer
                else:
                    # Fallback: just take the first substantial paragraph
                    answer_text = unique_paragraphs[0] if unique_paragraphs else ""
            else:
                answer_text = '\n\n'.join(unique_paragraphs) if unique_paragraphs else ""

        logger.info(f"RAG answer length: {len(answer_text)} chars")
        logger.info(f"Retrieved {len(nodes)} source chunks.")

        return RAGResult(
            answer=answer_text,
            nodes=nodes,
        )
    finally:
        # Try to close the vector store's async client if possible
        try:
            if hasattr(vector_store, 'close'):
                vector_store.close()
        except Exception:
            pass  # Ignore errors during cleanup

def run_rag_evaluation(
    config: RAGConfig,
    input_xlsx: str | os.PathLike,
    output_xlsx: str | os.PathLike,
    logger: logging.Logger,
) -> None:
    """
    Run RAG over a dataset stored in an XLSX file and write the results out.

    The input XLSX must contain at least these columns:
        - question
        - answer
        - relevant_doc_1
        - relevant_doc_2

    The output XLSX will contain all original columns plus:
        - rag_answer
        - rag_source_urls  (semicolon-separated list of retrieved URLs)
    """
    input_xlsx = os.fspath(input_xlsx)
    output_xlsx = os.fspath(output_xlsx)

    logger.info(f"Loading evaluation dataset from {input_xlsx!r}")

    try:
        df = pd.read_excel(input_xlsx)
    except Exception as e:
        logger.error(f"Failed to read XLSX {input_xlsx!r}: {e}")
        raise

    required_cols = ["question", "answer", "relevant_doc_1", "relevant_doc_2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input XLSX is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    n_rows = len(df)
    logger.info(f"Loaded {n_rows} evaluation rows")

    rag_answers: list[str] = []
    rag_source_urls: list[str] = []

    # Write progress incrementally (every 10 questions) for monitoring
    progress_file = Path(str(output_xlsx) + ".progress")

    for idx, row in df.iterrows():
        q = str(row["question"]).strip() if not pd.isna(row["question"]) else ""

        if not q:
            logger.warning(f"Row {idx}: empty question, skipping RAG call")
            rag_answers.append("")
            rag_source_urls.append("")
            continue

        logger.info(f"[{idx + 1}/{n_rows}] Running RAG for question: {q!r}")

        try:
            result = run_rag_query(config, q, logger)
        except Exception as e:
            logger.error(f"RAG query failed for row {idx}: {e}")
            rag_answers.append("")
            rag_source_urls.append("")
            continue

        # Store answer
        rag_answers.append(result.answer)

        # Collect URLs from retrieved nodes (unique, order-preserving)
        urls: list[str] = []
        for node in result.nodes:
            meta = getattr(node.node, "metadata", None) or {}
            url = meta.get("url")
            if url and url not in urls:
                urls.append(str(url))

        rag_source_urls.append("; ".join(urls))

        # Write progress every 10 questions
        if (idx + 1) % 10 == 0 or (idx + 1) == n_rows:
            df_temp = df.copy()
            df_temp["rag_answer"] = rag_answers + [""] * (len(df) - len(rag_answers))
            df_temp["rag_source_urls"] = rag_source_urls + [""] * (len(df) - len(rag_source_urls))
            try:
                df_temp.to_excel(progress_file, index=False)
                logger.info(f"Progress saved: {idx + 1}/{n_rows} questions completed")
            except Exception as e:
                logger.warning(f"Could not save progress: {e}")

    df["rag_answer"] = rag_answers
    df["rag_source_urls"] = rag_source_urls

    logger.info(f"Writing evaluation results to {output_xlsx!r}")
    try:
        df.to_excel(output_xlsx, index=False)
        # Clean up progress file
        if progress_file.exists():
            progress_file.unlink()
    except Exception as e:
        logger.error(f"Failed to write XLSX {output_xlsx!r}: {e}")
        raise

    logger.info("RAG evaluation finished successfully")

