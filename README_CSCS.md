# WARC Tools – Modular Pipeline for RAG and Baseline LLM Evaluation (CSCS)

> **NOTE**  
> The evaluation logic in this project originates from a modular pipeline design.  
> Evaluation is **integrated directly into the RAG pipeline**, while a separate **baseline module**
> is provided to evaluate base (non-RAG) models under the same evaluation setup.
>
> This README focuses on describing the **pipeline structure and module responsibilities**.
> For detailed evaluation methodology and metrics, please refer to the project-level documentation
> in `README.md` or the root documentation where applicable.

---

## Overview

This repository implements an **end-to-end pipeline** for large-scale web data ingestion and
evaluation of language models with and without retrieval augmentation.

The pipeline supports:

- Extraction of text data from **WARC files**
- Post-processing and **deduplication**
- **Embedding index construction**
- **Retrieval-Augmented Generation (RAG)** with built-in evaluation
- **Baseline LLM evaluation** without retrieval for comparison

The system is designed to run on **CSCS infrastructure** using a **single container image**
(Docker + Enroot / EDF).

---

## High-Level Architecture

```
WARCs
  ↓
Extractor        (HTML / PDF → JSONL)
  ↓
Processing       (Deduplication, normalization)
  ↓
Indexer          (JSONL → embedding index)
  ↓
RAG Pipeline     (retrieval + LLM + evaluation)

Baseline Pipeline
  ↓
LLM only (no retrieval)
  ↓
Evaluation
```

---

## Repository Structure

```
.
├── Dockerfile
├── README_CSCS.md
├── pyproject.toml
└── src/
    └── warc_tools/
        ├── __init__.py
        ├── extractor/
        ├── processing/
        ├── indexer/
        ├── rag/
        └── baseline/
```

---

## Module Descriptions

### extractor/ — WARC → JSONL

Converts raw web archives into structured text data.

- Reads `.warc` / `.warc.gz`
- Filters URLs by allowed domains
- Extracts HTML and PDF text
- Outputs JSONL
- Supports sharding for parallel execution

```bash
python -m warc_tools.extractor.cli --help
```

---

### processing/ — Post-processing

Improves corpus quality before indexing.

- Text normalization
- Content deduplication via hashing
- Streaming-friendly processing

```bash
python -m warc_tools.processing.cli --help
```

---

### indexer/ — Embedding Index

Builds a persistent retrieval index.

- Loads processed documents
- Computes embeddings in batches
- Persists vector index and metadata

```bash
python -m warc_tools.indexer.cli --help
```

---

### rag/ — Retrieval-Augmented Generation (with Evaluation)

End-to-end RAG pipeline.

- Query embedding
- Top-K retrieval
- Prompt construction
- LLM inference
- Integrated evaluation loop

```bash
python -m warc_tools.rag.cli --help
```

---

### baseline/ — Base LLM Evaluation (No RAG)

Evaluates base LLMs without retrieval.

- Same evaluation setup as RAG
- No indexing or retrieval
- Serves as control baseline

```bash
python -m warc_tools.baseline.cli --help
```

---

## Baseline vs RAG

| Aspect        | Baseline                | RAG                         |
|--------------|-------------------------|-----------------------------|
| Retrieval     | ❌ No                  | ✅ Yes                      |
| Index usage  | ❌ No                  | ✅ Yes                      |
| Evaluation   | ✅ Yes                 | ✅ Yes (built-in)           |
| Purpose      | Raw LLM performance     | Retrieval impact            |

---

## CSCS Compatibility

- Single container image
- Enroot / EDF-based execution
- Slurm-friendly
- CPU/GPU agnostic

See `README_CSCS.md` for CSCS-specific instructions.
