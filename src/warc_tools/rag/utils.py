# src/warc_tools/rag/utils.py
from __future__ import annotations

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM

from warc_tools.indexer.utils import require_env, setup_logging, get_embedding_model_from_env


def load_env_if_dev() -> None:
    """Load .env if DEV_MODE=1/true/yes."""
    #dev_mode = os.getenv("DEV_MODE", "0").lower() in ("1", "true", "yes")
    dev_mode = 1
    if dev_mode:
        load_dotenv()

def get_llm_from_env(logger: logging.Logger):
    """
    Build an LLM instance based on env vars:

    LLM_PROVIDER:
      - "ollama" : local Ollama
      - "cscs"   : OpenAI-compatible endpoint (e.g. SwissAI)
      - "openai" : OpenAI / OpenAI-compatible
      - "hf"     : Hugging Face local / hub

    LLM_MODEL:      model name
    LLM_BASE_URL:   (for cscs/openai) base URL
    LLM_API_KEY:    (for cscs/openai) token
    LLM_TEMPERATURE: float, default 0.1
    LLM_MAX_TOKENS: int,   default 1024
    """
    provider = require_env("LLM_PROVIDER").lower()
    model = require_env("LLM_MODEL")

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    logger.info(f"Using LLM provider='{provider}', model='{model}'")

    if provider == "ollama":
        base_url = os.getenv("LLM_BASE_URL")  # optional, defaults to 127.0.0.1
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "request_timeout": int(os.getenv("LLM_TIMEOUT", "60")),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return Ollama(**kwargs)

    elif provider in ("openai", "cscs"):
        base_url = require_env("LLM_BASE_URL")
        api_key = require_env("LLM_API_KEY")

        # For CSCS, monkey-patch validation to allow custom model names
        if provider == "cscs":
            import llama_index.llms.openai.utils as openai_utils
            from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS

            # Add model to ALL_AVAILABLE_MODELS before any validation
            if model not in ALL_AVAILABLE_MODELS:
                ALL_AVAILABLE_MODELS[model] = 8192  # Default context size

            # Patch is_chatcomp_api_supported to always return True for our model
            original_is_chatcomp = openai_utils.is_chatcomp_api_supported
            def patched_is_chatcomp(model_name: str) -> bool:
                if model_name == model:
                    return True
                return original_is_chatcomp(model_name)
            
            openai_utils.is_chatcomp_api_supported = patched_is_chatcomp

        return OpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider in ("hf", "huggingface"):
        # Hugging Face LLM (local or via transformers) – adjust kwargs as needed
        return HuggingFaceLLM(
            model_name=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

    else:
        logger.error(f"Unsupported LLM_PROVIDER: {provider}")
        sys.exit(1)
