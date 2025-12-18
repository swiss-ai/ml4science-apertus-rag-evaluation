"""
Baseline CLI Module

Command-line interface for running baseline LLM evaluations on Excel datasets.
Processes questions through CSCS-hosted models without RAG context.

Usage:
    python -m warc_tools.baseline.cli input.xlsx output.xlsx MODEL_NAME
"""
# src/warc_tools/baseline/cli.py
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import logging

import pandas as pd
import openai  # CSCS endpoint is OpenAI-compatible

DEFAULT_BASE_URL = "https://api.swissai.cscs.ch/v1"

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("warc_tools.baseline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if main() is called twice in tests
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            )
            logger.addHandler(fh)

    return logger

def load_env_if_dev() -> None:
    """Optional: mimic your rag utils; safe no-op in prod."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()

def get_cscs_client(base_url: str, api_key: str) -> openai.Client:
    """
    Create an OpenAI-compatible client for the CSCS endpoint.
    """
    return openai.Client(api_key=api_key, base_url=base_url)


def call_baseline_model(
    client: openai.Client,
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 3.0,
) -> str:
    """
    Call the baseline model on a single question and return the answer text.

    Uses non-streaming API for simplicity.
    Retries a few times on errors.
    """
    if not question.strip():
        return ""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            # OpenAI-compatible response
            if not resp.choices:
                return ""
            answer = resp.choices[0].message.content or ""
            return answer.strip()
        except Exception as e:  # You may want more granular handling
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

    # Should never reach here
    if last_error:
        raise last_error
    return ""


def usage() -> None:
    print(
        "Usage:\n"
        "  python -m warc_tools.baseline.cli input.xlsx output.xlsx MODEL_NAME\n\n"
        "Example:\n"
        "  python -m warc_tools.baseline.cli data/gold.xlsx data/baseline.xlsx llama-3-70b-instruct\n",
        file=sys.stderr,
    )


def main() -> None:
    """
    CLI to run a CSCS-hosted baseline model over an Excel file of questions.

    Input Excel must have at least a 'question' column.
    All original columns are preserved; two new columns are appended:
        - baseline_answer
        - baseline_model

    Environment variables:
        CSCS_API_KEY   (required)
        CSCS_BASE_URL  (optional, default: https://api.swissai.cscs.ch/v1)
        LOG_LEVEL      (optional, default: INFO)
        LOG_FILE       (optional, log to file if set)
    """
    load_env_if_dev()

    if len(sys.argv) < 4:
        usage()
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model_name = sys.argv[3]

    # Logging setup
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE") or None
    logger = setup_logging(log_level, log_file)

    logger.info(f"Baseline evaluation: input={input_path!s}, output={output_path!s}")
    logger.info(f"Using baseline model: {model_name}")

    # CSCS API config
    api_key = require_env("CSCS_API_KEY")
    base_url = os.getenv("CSCS_BASE_URL", DEFAULT_BASE_URL)

    client = get_cscs_client(base_url=base_url, api_key=api_key)

    # Read Excel
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        logger.error(f"Failed to read Excel {input_path!s}: {e}")
        sys.exit(1)

    if "question" not in df.columns:
        logger.error(
            f"Input Excel must contain a 'question' column. "
            f"Found columns: {list(df.columns)}"
        )
        sys.exit(1)

    n_rows = len(df)
    logger.info(f"Loaded {n_rows} rows from {input_path!s}")

    # Prepare output columns
    baseline_answers: list[str] = []
    baseline_models: list[str] = []

    # Optional: keep an existing baseline_answer if already present and non-empty
    skip_existing = os.getenv("BASELINE_SKIP_EXISTING", "1") == "1"

    for idx, row in df.iterrows():
        question = str(row["question"]) if not pd.isna(row["question"]) else ""

        if skip_existing and "baseline_answer" in df.columns:
            existing = row.get("baseline_answer", "")
            if isinstance(existing, str) and existing.strip():
                baseline_answers.append(existing)
                baseline_models.append(row.get("baseline_model", model_name))
                logger.debug(f"Row {idx}: skipping, baseline_answer already present")
                continue

        if not question.strip():
            logger.warning(f"Row {idx}: empty question, writing empty baseline answer")
            baseline_answers.append("")
            baseline_models.append(model_name)
            continue

        logger.info(f"[{idx + 1}/{n_rows}] Calling baseline model for question...")
        try:
            answer = call_baseline_model(
                client=client,
                model=model_name,
                question=question,
                # You can add a system prompt here if you want
                system_prompt=os.getenv("BASELINE_SYSTEM_PROMPT"),
                temperature=float(os.getenv("BASELINE_TEMPERATURE", "0.0")),
            )
        except Exception as e:
            logger.error(f"Row {idx}: baseline model call failed: {e}")
            answer = ""

        baseline_answers.append(answer)
        baseline_models.append(model_name)

        time.sleep(float(os.getenv("BASELINE_SLEEP_SECONDS", "1.0")))

    # Attach / overwrite columns
    df["baseline_answer"] = baseline_answers
    df["baseline_model"] = baseline_models

    # Write to output Excel
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
    except Exception as e:
        logger.error(f"Failed to write Excel {output_path!s}: {e}")
        sys.exit(1)

    logger.info(f"Baseline Excel written to {output_path!s}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
