"""
Indexer Module

Main indexing logic for processing JSONL files and indexing them into Elasticsearch.
Handles document chunking, embedding generation, and batch indexing with vector search support.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .utils import batched, create_es_index_for_vectors, get_embedding_model_from_env
from .config import IndexerConfig

def _parse_csv_set(v: str | None) -> Set[str]:
    if not v:
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}

def _parse_years(v: str | None) -> Set[int]:
    if not v:
        return set()
    out: Set[int] = set()
    for x in v.split(","):
        x = x.strip()
        if x.isdigit():
            out.add(int(x))
    return out

def _load_completed_files(path: Path, logger: logging.Logger) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except Exception as e:
        logger.warning("Could not read completed files list %s: %s", path, e)
        return set()

def _append_completed_file(path: Path, file_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(file_path) + "\n")


def _detect_layout(jsonl_root: Path) -> Tuple[Path, ...]:
    """If jsonl_root contains html/ or pdf/, index under those; else index jsonl_root itself."""
    candidates: list[Path] = []
    for name in ("html", "pdf"):
        p = jsonl_root / name
        if p.exists() and p.is_dir():
            candidates.append(p)
    return tuple(candidates) if candidates else (jsonl_root,)

def _iter_jsonl_files(jsonl_root: Path) -> Iterator[Path]:
    """Recursively yield *.jsonl files."""
    yield from sorted(p for p in jsonl_root.rglob("*.jsonl") if p.is_file())

def _infer_modality_and_year(path: Path) -> tuple[str | None, int | None]:
    """Infer modality from path parts (html/pdf) and year from filename 'YYYY.jsonl'."""
    modality = None
    parts = {p.lower() for p in path.parts}
    if "html" in parts:
        modality = "html"
    elif "pdf" in parts:
        modality = "pdf"

    year = None
    stem = path.stem
    if stem.isdigit() and len(stem) == 4:
        year = int(stem)

    return modality, year

def iter_documents_from_jsonl(jsonl_dir: Path, logger: logging.Logger) -> Iterator[Document]:
    roots = _detect_layout(jsonl_dir)
    jsonl_files: list[Path] = []
    for root in roots:
        jsonl_files.extend(list(_iter_jsonl_files(root)))

    # Optional filters
    years_filter = _parse_years(os.getenv("YEARS"))  # e.g. "2025" or "2024,2025"
    modalities_filter = _parse_csv_set(os.getenv("MODALITIES"))  # e.g. "html" or "pdf" or "html,pdf"

    # Optional checkpointing
    skip_completed = os.getenv("SKIP_COMPLETED_FILES", "0").lower() in ("1", "true", "yes")
    completed_path = Path(os.getenv("COMPLETED_FILES_FILE", "logs/completed_jsonl_files.txt"))
    completed = _load_completed_files(completed_path, logger) if skip_completed else set()

    logger.info("Index input roots: %s", ", ".join(str(r) for r in roots))
    logger.info("Found %d JSONL files under %s", len(jsonl_files), jsonl_dir)
    if years_filter:
        logger.info("Filter YEARS=%s", ",".join(map(str, sorted(years_filter))))
    if modalities_filter:
        logger.info("Filter MODALITIES=%s", ",".join(sorted(modalities_filter)))
    if skip_completed:
        logger.info("Skipping completed files from %s (count=%d)", completed_path, len(completed))
    
    iter_documents_from_jsonl.current_file = None

    for file_idx, jsonl_path in enumerate(jsonl_files, start=1):
        modality, year_from_path = _infer_modality_and_year(jsonl_path)

        # apply filters
        if modalities_filter and (modality or "") not in modalities_filter:
            continue
        if years_filter and (year_from_path not in years_filter):
            continue

        if skip_completed and str(jsonl_path) in completed:
            continue

        iter_documents_from_jsonl.current_file = str(jsonl_path)
        logger.info("[INPUT %d/%d] READING %s", file_idx, len(jsonl_files), jsonl_path)

        seen = kept = 0
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_idx, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                seen += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("JSON decode error: %s line=%d (skipping)", jsonl_path.name, line_idx)
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
                    "modality": modality,
                    "year": year_from_path,
                    "_src_file": str(jsonl_path),
                    "_src_line": line_idx,
                }

                kept += 1
                yield Document(text=text, metadata=metadata)

        logger.info("[INPUT %d/%d] READ DONE %s seen=%d kept=%d", file_idx, len(jsonl_files), jsonl_path.name, seen, kept)

        # mark file completed once fully read
        if skip_completed:
            _append_completed_file(completed_path, jsonl_path)
            completed.add(str(jsonl_path))

def _summarize_sources(batch: List[Document], top_n: int = 3) -> str:
    c = Counter((d.metadata or {}).get("jsonl_file", "unknown") for d in batch)
    items = ", ".join(f"{Path(k).name}:{v}" for k, v in c.most_common(top_n))
    return f"{len(c)} files ({items}{'...' if len(c) > top_n else ''})"

def _node_text(node) -> str:
    if hasattr(node, "get_content"):
        return node.get_content(metadata_mode="none")
    return getattr(node, "text", "")

def _stable_id(parts: List[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()

def _estimate_action_bytes(action: Dict[str, Any]) -> int:
    # Bulk is JSON lines; we approximate by JSON encoding the action dict once.
    # It’s a useful upper bound for “is this single doc absurdly large?”
    return len(json.dumps(action, ensure_ascii=False).encode("utf-8"))

def run_indexing(config: IndexerConfig, logger: logging.Logger) -> None:
    if not config.jsonl_dir.exists():
        logger.error("JSONL directory does not exist: %s", config.jsonl_dir)
        raise FileNotFoundError(config.jsonl_dir)

    # Bulk safety limits
    max_chunk_mb = float(os.getenv("ES_BULK_MAX_MB", "1"))
    max_chunk_bytes = int(max_chunk_mb * 1024 * 1024)
    bulk_actions_chunk_size = int(os.getenv("ES_BULK_ACTIONS", "200"))
    
    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "0"))  # 0 = no truncation

    logger.info("=== Indexer configuration ===")
    logger.info("JSONL_DIR:                  %s", config.jsonl_dir)
    logger.info("ES_URL:                     %s", config.es_url)
    logger.info("ES_INDEX_NAME:              %s", config.index_name)
    logger.info("BATCH_SIZE:                 %d", config.batch_size)
    logger.info("CHUNK_SIZE:                 %d", config.chunk_size)
    logger.info("CHUNK_OVERLAP:              %d", config.chunk_overlap)
    logger.info("FAILED_WARCS_INDEXING_FILE: %s", config.failed_warcs_file)
    logger.info("NO_RECREATE_INDEX:          %s", config.no_recreate_index)
    logger.info("ES_BULK_MAX_MB:             %.2f", max_chunk_mb)
    logger.info("ES_BULK_ACTIONS:            %d", bulk_actions_chunk_size)
    logger.info("MAX_TEXT_CHARS:             %d", max_text_chars)
    logger.info("============================")

    # Build ES client (compression)
    es_kwargs = dict(
        verify_certs=config.es_verify_certs,
        http_compress=True,
        request_timeout=120,
    )
    if config.es_user and config.es_password:
        es_client = Elasticsearch(
            config.es_url,
            basic_auth=(config.es_user, config.es_password),
            **es_kwargs,
        )
    else:
        es_client = Elasticsearch(config.es_url, **es_kwargs)

    logger.info("Pinging Elasticsearch...")
    if not es_client.ping():
        raise RuntimeError(f"Elasticsearch ping failed: {config.es_url}")
    logger.info("Elasticsearch is reachable.")

    embed_model = get_embedding_model_from_env(logger)

    logger.info("Testing embedding call to determine dimension...")
    dim = len(embed_model.get_text_embedding("hello world"))
    logger.info("Embedding backend OK, dimension=%d", dim)

    if config.no_recreate_index:
        logger.info("NO_RECREATE_INDEX=1 -> skipping index recreation.")
        if not es_client.indices.exists(index=config.index_name):
            logger.info("Index '%s' does not exist; creating...", config.index_name)
            create_es_index_for_vectors(es_client, config.index_name, dim, logger)
    else:
        create_es_index_for_vectors(es_client, config.index_name, dim, logger)

    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    docs_iter = iter_documents_from_jsonl(config.jsonl_dir, logger)
    failed_warcs: Set[str] = set()

    batch_idx = 0
    total_input_docs = 0
    total_nodes = 0
    total_indexed = 0
    start_all = time.time()

    progress_every = int(os.getenv("PROGRESS_EVERY", "200"))
    last_t = start_all
    last_docs = 0
    last_nodes = 0

    def actions_for_nodes(nodes, parent_docs: List[Document]) -> Iterator[Dict[str, Any]]:
        nonlocal total_nodes

        for i, node in enumerate(nodes):
            text = _node_text(node)
            if max_text_chars > 0 and len(text) > max_text_chars:
                text = text[:max_text_chars]
            
            meta = {}
            try:
                meta.update(getattr(node, "metadata", {}) or {})
            except Exception:
                pass

            url = meta.get("url") or meta.get("source_url") or ""
            capture = str(meta.get("capture_time") or "")
            src_file = str(meta.get("jsonl_file") or meta.get("_src_file") or "")
            src_line = str(meta.get("_src_line") or "")

            # A stable id helps avoid duplicates on reruns
            doc_id = _stable_id([url, capture, src_file, src_line, str(i)])

            # Embedding should already be attached
            emb = getattr(node, "embedding", None)
            if emb is None:
                # As a fallback, embed here (should not happen)
                emb = embed_model.get_text_embedding(text)

            source = {
                "text": text,
                "metadata": meta,
                "embedding": emb,
            }

            action = {
                "_op_type": "index",
                "_index": config.index_name,
                "_id": doc_id,
                "_source": source,
            }

            total_nodes += 1
            yield action

    for batch in batched(docs_iter, config.batch_size):
        batch_idx += 1
        total_input_docs += len(batch)

        logger.info(
            "[BATCH %d] INPUT docs=%d read_total=%d sources=%s",
            batch_idx, len(batch), total_input_docs, _summarize_sources(batch)
        )

        t0 = time.time()
        try:
            # Split into nodes/chunks
            nodes = splitter.get_nodes_from_documents(batch)
            logger.info("[BATCH %d] SPLIT nodes=%d", batch_idx, len(nodes))

            # Embed nodes in one batch call
            texts = [_node_text(n) for n in nodes]
            if max_text_chars > 0:
                texts = [t[:max_text_chars] for t in texts]
            
            vectors = embed_model.get_text_embedding_batch(texts)
            for n, v in zip(nodes, vectors):
                n.embedding = v

            logger.info("[BATCH %d] EMBED done vectors=%d", batch_idx, len(vectors))

            acts = actions_for_nodes(nodes, batch)

            ok_count = 0
            fail_count = 0
            bytes_too_big = 0

            # break up requests by both chunk_size (actions) and max_chunk_bytes (bytes)
            for ok, info in streaming_bulk(
                client=es_client,
                actions=acts,
                chunk_size=bulk_actions_chunk_size,
                max_chunk_bytes=max_chunk_bytes,
                raise_on_error=False,
                raise_on_exception=False,
                request_timeout=120,
                refresh=False,  # keep it off for bulk loads
            ):
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    # Detect single-document-too-large style failures
                    try:
                        op = next(iter(info.values()))
                        err = op.get("error")
                        if err:
                            pass
                    except Exception:
                        pass

            total_indexed += ok_count

            if batch_idx % progress_every == 0:
                now = time.time()
                dt = now - last_t
                total_dt = now - start_all
                docs_rate = (total_input_docs - last_docs) / dt if dt > 0 else 0.0
                nodes_rate = (total_indexed - last_nodes) / dt if dt > 0 else 0.0
                curr_file = getattr(iter_documents_from_jsonl, "current_file", None)
                logger.info(
                    "[PROGRESS] batches=%d input_docs=%d indexed_nodes=%d rate_docs=%.1f/s rate_nodes=%.1f/s elapsed=%.1fmin current_file=%s",
                    batch_idx, total_input_docs, total_indexed, docs_rate, nodes_rate, total_dt / 60.0,
                    curr_file or "unknown",
                )
                last_t = now
                last_docs = total_input_docs
                last_nodes = total_indexed

            logger.info(
                "[BATCH %d] BULK ok=%d failed=%d (chunk_limit=%.2fMB actions_limit=%d)",
                batch_idx, ok_count, fail_count, max_chunk_mb, bulk_actions_chunk_size
            )

            if fail_count > 0:
                # Mark WARCs touched by this batch
                for doc in batch:
                    sw = (doc.metadata or {}).get("source_warc")
                    if sw:
                        failed_warcs.add(sw)

        except Exception as e:
            logger.error("[BATCH %d] FAILED err=%s (continuing)", batch_idx, e)
            for doc in batch:
                sw = (doc.metadata or {}).get("source_warc")
                if sw:
                    failed_warcs.add(sw)
            continue

        logger.info("[BATCH %d] DONE time=%.1fs", batch_idx, time.time() - t0)

    logger.info(
        "Finished indexing: input_docs=%d nodes_created=%d docs_indexed=%d total_time=%.1fs",
        total_input_docs, total_nodes, total_indexed, time.time() - start_all
    )

    if failed_warcs:
        logger.warning(
            "Failed batches touched %d WARCs; writing: %s",
            len(failed_warcs), config.failed_warcs_file
        )
        config.failed_warcs_file.parent.mkdir(parents=True, exist_ok=True)
        with config.failed_warcs_file.open("w", encoding="utf-8") as f:
            for w in sorted(failed_warcs):
                f.write(w + "\n")
    else:
        logger.info("No WARCs failed during indexing.")
