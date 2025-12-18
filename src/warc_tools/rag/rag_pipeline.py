from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, List

import pandas as pd
from elasticsearch import Elasticsearch

from warc_tools.indexer.utils import get_embedding_model_from_env
from .utils import get_cscs_llm_from_env


@dataclass
class RAGConfig:
    es_url: str
    index_name: str
    es_user: str | None = None
    es_password: str | None = None
    top_k: int = 5

@dataclass
class SimpleNode:
    text: str
    metadata: dict

    def get_content(self, metadata_mode: str = "none") -> str:
        return self.text


@dataclass
class SimpleNodeWithScore:
    node: SimpleNode
    score: float


@dataclass
class RAGResult:
    answer: str
    nodes: List[SimpleNodeWithScore]


def _es_client(config: RAGConfig) -> Elasticsearch:
    if config.es_user and config.es_password:
        return Elasticsearch(config.es_url, basic_auth=(config.es_user, config.es_password))
    return Elasticsearch(config.es_url)


def _merge_metadata(src: dict) -> dict:
    """
    Supports both schemas:
      A) nested metadata: {"text": "...", "metadata": {...}}
      B) flattened metadata: {"text": "...", "url": "...", ...}
    Returns a metadata dict suitable for printing (url/source_warc/capture_time...).
    """
    meta: dict[str, Any] = {}

    md = src.get("metadata")
    if isinstance(md, dict):
        meta.update(md)

    # bring in any top-level fields not already present
    for k, v in src.items():
        if k in ("text", "embedding", "metadata"):
            continue
        if k not in meta:
            meta[k] = v

    return meta


def _retrieve_topk(
    config: RAGConfig,
    query: str,
    logger: logging.Logger,
) -> List[SimpleNodeWithScore]:
    embed_model = get_embedding_model_from_env(logger)
    qvec = embed_model.get_query_embedding(query)

    if not isinstance(qvec, list) or not qvec:
        raise RuntimeError("Query embedding is empty/invalid.")
    if len(qvec) != 1024:
        raise RuntimeError(f"Query embedding dims={len(qvec)} but index expects 1024.")

    es = _es_client(config)

    resp: dict[str, Any] = es.search(
        index=config.index_name,
        size=config.top_k,
        _source=True,
        knn={
            "field": "embedding",
            "query_vector": qvec,
            "k": config.top_k,
            "num_candidates": max(100, config.top_k * 20),
        },
    )

    hits = (resp.get("hits") or {}).get("hits") or []
    out: List[SimpleNodeWithScore] = []

    for h in hits:
        src = h.get("_source") or {}

        text = src.get("text")
        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "Skipping hit with missing/empty text: _id=%s _source_keys=%s",
                h.get("_id"),
                sorted(src.keys()),
            )
            continue

        meta = _merge_metadata(src)

        # debug if metadata is unexpectedly empty
        if not meta.get("url") and not meta.get("capture_time") and not meta.get("source_warc"):
            logger.debug(
                "Hit has text but no expected metadata fields. _id=%s meta_keys=%s src_keys=%s",
                h.get("_id"),
                sorted(meta.keys()),
                sorted(src.keys()),
            )

        out.append(
            SimpleNodeWithScore(
                node=SimpleNode(text=text, metadata=meta),
                score=float(h.get("_score") or 0.0),
            )
        )

    return out


def _format_context_from_nodes(nodes: List[SimpleNodeWithScore]) -> str:
    parts: list[str] = []
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        title = meta.get("url") or meta.get("title") or f"chunk-{i}"
        capture = meta.get("capture_time")
        warc = meta.get("source_warc")

        header_bits = [str(title)]
        if capture:
            header_bits.append(f"capture_time={capture}")
        if warc:
            header_bits.append(f"warc={warc}")

        text = n.node.get_content(metadata_mode="none")
        parts.append(f"Source {i}: " + " | ".join(header_bits) + "\n" + text)

    return "\n\n".join(parts)


def _call_cscs_llm_with_context(
    question: str,
    nodes: List[SimpleNodeWithScore],
    logger: logging.Logger,
) -> str:
    client, model = get_cscs_llm_from_env(logger)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    context = _format_context_from_nodes(nodes)

    logger.info("Context chars=%d; first 200=%r", len(context), context[:200])

    logger.info("[LLM CONTEXT PREVIEW] %r", context)

    prompt = (
        "You answer questions about ETH Zurich web pages.\n"
        "Use the Context as your only source of information.\n"
        "If the Context contains relevant information, answer concisely and quote/cite the supporting snippet.\n"
        "If the Context does not contain relevant information, say: \"I don't know based on the provided sources.\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=False,
    )

    if not resp.choices:
        return ""

    return (resp.choices[0].message.content or "").strip()


def run_rag_query(
    config: RAGConfig,
    query: str,
    logger: logging.Logger,
) -> RAGResult:
    logger.info("Retrieving top-k chunks via Elasticsearch kNN (bypassing LlamaIndex parsing)")
    logger.info("Using ES index=%s url=%s top_k=%d", config.index_name, config.es_url, config.top_k)

    nodes = _retrieve_topk(config, query, logger)
    logger.info("Retrieved %d source chunks.", len(nodes))

    answer = _call_cscs_llm_with_context(query, nodes, logger)
    logger.info("RAG answer length: %d chars", len(answer))

    return RAGResult(answer=answer, nodes=nodes)


def run_rag_evaluation(
    config: RAGConfig,
    input_xlsx: str | os.PathLike,
    output_xlsx: str | os.PathLike,
    logger: logging.Logger,
) -> None:
    input_xlsx = os.fspath(input_xlsx)
    output_xlsx = os.fspath(output_xlsx)

    df = pd.read_excel(input_xlsx)

    required_cols = ["question", "answer", "relevant_doc_1", "relevant_doc_2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input XLSX is missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    n_rows = len(df)
    rag_answers: list[str] = []
    rag_source_urls: list[str] = []

    for idx, row in df.iterrows():
        q = str(row["question"]).strip() if not pd.isna(row["question"]) else ""
        if not q:
            logger.warning("Row %d: empty question, skipping", idx)
            rag_answers.append("")
            rag_source_urls.append("")
            continue

        logger.info("[%d/%d] RAG question: %r", idx + 1, n_rows, q)

        try:
            result = run_rag_query(config, q, logger)
        except Exception as e:
            logger.error("RAG query failed for row %d: %s", idx, e)
            rag_answers.append("")
            rag_source_urls.append("")
            continue

        rag_answers.append(result.answer)

        urls: list[str] = []
        for nws in result.nodes:
            meta = nws.node.metadata or {}
            url = meta.get("url")
            if url and url not in urls:
                urls.append(str(url))
        rag_source_urls.append("; ".join(urls))

    df["rag_answer"] = rag_answers
    df["rag_source_urls"] = rag_source_urls
    df.to_excel(output_xlsx, index=False)