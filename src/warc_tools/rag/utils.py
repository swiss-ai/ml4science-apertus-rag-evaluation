# src/warc_tools/rag/utils.py
from __future__ import annotations

import logging
import os
import sys
from typing import Tuple

from dotenv import load_dotenv
import openai

from warc_tools.indexer.utils import require_env


def load_env_if_dev() -> None:
    """
    Load .env only in development (ENV=dev or unset).

    This mirrors what you already did before.
    """
    env = os.getenv("ENV", "dev").lower()
    if env == "dev":
        load_dotenv()


def get_cscs_llm_from_env(logger: logging.Logger) -> Tuple[openai.Client, str]:
    """
    Configure a CSCS-hosted LLM via OpenAI-compatible API.

    Only supports:
        LLM_PROVIDER = "cscs"

    Required env vars:
        LLM_PROVIDER   = "cscs"
        LLM_BASE_URL   = OpenAI-compatible base URL (e.g. https://api.swissai.cscs.ch/v1)
        LLM_API_KEY    = API key / token for CSCS
        LLM_MODEL      = model name (e.g. Qwen/Qwen3-8B)

    Optional:
        LLM_TEMPERATURE
    """
    provider = require_env("LLM_PROVIDER").lower()
    if provider != "cscs":
        logger.error(
            f"Only LLM_PROVIDER='cscs' is supported for warc_tools.rag now, "
            f"got {provider!r}"
        )
        sys.exit(1)

    base_url = require_env("LLM_BASE_URL")
    api_key = require_env("LLM_API_KEY")
    model = require_env("LLM_MODEL")

    logger.info(f"Using CSCS LLM provider: model='{model}', base_url='{base_url}'")

    client = openai.Client(api_key=api_key, base_url=base_url)
    return client, model
