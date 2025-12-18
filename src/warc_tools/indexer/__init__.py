"""
Indexer Module

This module provides functionality to index extracted content into Elasticsearch
with embeddings for vector search. Supports chunking, embedding generation, and
batch indexing operations.

Main Components:
    - IndexerConfig: Configuration for indexing process
    - run_indexing(): Main indexing function
"""
# src/warc_tools/indexer/__init__.py
from .indexer import IndexerConfig, run_indexing

__all__ = ["IndexerConfig", "run_indexing"]
