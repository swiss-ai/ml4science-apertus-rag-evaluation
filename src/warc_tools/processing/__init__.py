"""
Processing Module

This module provides content deduplication functionality to remove duplicate
content from extracted JSONL files using content hashing.

Main Components:
    - DedupConfig: Configuration for deduplication process
    - run_dedup(): Main deduplication function
    - run_dedup_from_env(): Run deduplication from environment variables
"""
from __future__ import annotations

__all__ = ["run_dedup_from_env"]

from .cli import run_dedup_from_env
