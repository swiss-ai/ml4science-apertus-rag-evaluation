"""
RAG Module

This module provides RAG (Retrieval-Augmented Generation) functionality for querying
ETH Zurich web archive documents using Elasticsearch as the vector store.

Main Components:
    - RAGConfig: Configuration for RAG pipeline
    - RAGResult: Result containing answer and retrieved nodes
    - run_rag_query(): Execute a single RAG query
"""
# src/warc_tools/rag/__init__.py
from .rag_pipeline import RAGConfig, RAGResult, run_rag_query

__all__ = ["RAGConfig", "RAGResult", "run_rag_query"]
