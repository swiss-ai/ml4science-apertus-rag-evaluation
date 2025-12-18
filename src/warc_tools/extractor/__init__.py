"""
WARC Extractor Module

This module provides functionality to extract content from WARC files.
Supports extraction of HTML and PDF content from web archives.

Main Components:
    - ExtractorConfig: Configuration for extraction process
    - run_extraction(): Main extraction function
"""
# src/extractor/__init__.py
from .warc_extractor import ExtractorConfig, run_extraction

__all__ = ["ExtractorConfig", "run_extraction"]
