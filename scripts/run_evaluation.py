#!/usr/bin/env python3
"""
Run baseline evaluation for a single model.

Usage:
    python scripts/run_evaluation.py --model <model_name> [--api <provider>] [--output <output_file>]

Examples:
    # CSCS self-hosted model
    python scripts/run_evaluation.py --model swiss-ai/Apertus-8B-Instruct-2509

    # Cloud model via API
    python scripts/run_evaluation.py --model claude-sonnet-4.5 --api anthropic
    python scripts/run_evaluation.py --model gpt-4 --api openai
    python scripts/run_evaluation.py --model gemini-2.0 --api google
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from dotenv import load_dotenv

# Optional imports for cloud APIs
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reproducibility settings
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 500
RANDOM_SEED = 42


def load_questions(json_path: Path) -> list[Dict[str, Any]]:
    """Load questions from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_cscs_model(
    client: openai.Client,
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Call a CSCS-hosted model via OpenAI-compatible API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=RANDOM_SEED if RANDOM_SEED else None,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling CSCS model: {e}", file=sys.stderr)
        raise


def call_anthropic_model(
    client: Anthropic,
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Anthropic Claude model (tools disabled)."""
    messages = [{"role": "user", "content": question}]
    
    try:
        resp = client.messages.create(
            model=model,
            messages=messages,
            system=system_prompt or "",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            tools=[],  # CRITICAL: Disable tools/web search
        )
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        print(f"Error calling Anthropic model: {e}", file=sys.stderr)
        raise


def call_openai_model(
    client: openai.Client,
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Call OpenAI model (function calling disabled)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=RANDOM_SEED if RANDOM_SEED else None,
            # CRITICAL: No function calling enabled
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling OpenAI model: {e}", file=sys.stderr)
        raise


def call_google_model(
    client: Any,  # genai.Client when available
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Google Gemini model (grounding/search disabled)."""
    # Note: Google API structure may vary, adjust as needed
    try:
        # CRITICAL: Disable grounding and search
        resp = client.models.generate_content(
            model=model,
            contents=question,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_TOKENS,
            },
            # No grounding_config or search settings
        )
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        print(f"Error calling Google model: {e}", file=sys.stderr)
        raise


def run_evaluation(
    model_name: str,
    api_provider: Optional[str] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Run evaluation for a single model."""
    # Load questions
    questions_path = Path(__file__).parent.parent / "test_set" / "eth_questions_100.json"
    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Determine API provider
    if api_provider is None:
        # Check if it's a CSCS model (default)
        if "swiss-ai" in model_name or "/" in model_name:
            api_provider = "cscs"
        else:
            # Try to infer from model name
            if "claude" in model_name.lower():
                api_provider = "anthropic"
            elif "gpt" in model_name.lower():
                api_provider = "openai"
            elif "gemini" in model_name.lower():
                api_provider = "google"
            elif "llama" in model_name.lower() and os.getenv("OLLAMA_BASE_URL"):
                api_provider = "ollama"
            else:
                api_provider = "cscs"  # Default to CSCS

    # Initialize API client
    if api_provider == "cscs":
        api_key = os.getenv("CSCS_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("CSCS_BASE_URL", "https://api.swissai.cscs.ch/v1")
        if not api_key:
            print("Error: CSCS_API_KEY or LLM_API_KEY environment variable required", file=sys.stderr)
            sys.exit(1)
        client = openai.Client(api_key=api_key, base_url=base_url)
        call_func = lambda q, sp=None: call_cscs_model(client, model_name, q, sp)
    elif api_provider == "anthropic":
        if Anthropic is None:
            print("Error: anthropic package not installed. Install with: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable required", file=sys.stderr)
            sys.exit(1)
        client = Anthropic(api_key=api_key)
        # Map model name if needed
        claude_model = model_name if "claude" in model_name.lower() else f"claude-{model_name}"
        call_func = lambda q, sp=None: call_anthropic_model(client, claude_model, q, sp)
    elif api_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable required", file=sys.stderr)
            sys.exit(1)
        client = openai.Client(api_key=api_key)
        call_func = lambda q, sp=None: call_openai_model(client, model_name, q, sp)
    elif api_provider == "google":
        if genai is None:
            print("Error: google-generativeai package not installed. Install with: pip install google-generativeai", file=sys.stderr)
            sys.exit(1)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable required", file=sys.stderr)
            sys.exit(1)
        client = genai.Client(api_key=api_key)
        call_func = lambda q, sp=None: call_google_model(client, model_name, q, sp)
    elif api_provider == "ollama":
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError:
            print("Error: llama-index-llms-ollama package not installed. Install with: pip install llama-index-llms-ollama", file=sys.stderr)
            sys.exit(1)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=TEMPERATURE,
            request_timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        )
        def call_ollama_func(question: str, system_prompt: Optional[str] = None) -> str:
            prompt = question
            if system_prompt:
                prompt = f"{system_prompt}\n\n{question}"
            response = llm.complete(prompt)
            return str(response) if response else ""
        call_func = call_ollama_func
    else:
        print(f"Error: Unknown API provider: {api_provider}", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    results = []
    system_prompt = os.getenv("BASELINE_SYSTEM_PROMPT")

    print(f"\nRunning evaluation for model: {model_name} (provider: {api_provider})")
    print(f"Settings: temperature={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_TOKENS}\n")

    for i, q_data in enumerate(questions, 1):
        question = q_data["question"]
        print(f"[{i}/{len(questions)}] Processing question {q_data['question_id']}...", end=" ", flush=True)

        try:
            response = call_func(question, system_prompt)
            results.append({
                "question_id": q_data["question_id"],
                "question": question,
                "language": q_data.get("language", "unknown"),
                "ground_truth": q_data.get("ground_truth", ""),
                "model_response": response,
                "model_name": model_name,
                "api_provider": api_provider,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
            })
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "question_id": q_data["question_id"],
                "question": question,
                "language": q_data.get("language", "unknown"),
                "ground_truth": q_data.get("ground_truth", ""),
                "model_response": f"ERROR: {str(e)}",
                "model_name": model_name,
                "api_provider": api_provider,
                "error": str(e),
            })

        # Rate limiting
        time.sleep(float(os.getenv("BASELINE_SLEEP_SECONDS", "1.0")))

    # Save results
    if output_file is None:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        model_safe = model_name.replace("/", "_").replace(" ", "_")
        output_file = results_dir / "baseline_evaluation" / f"{model_safe}_responses.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_file}")
    print(f"  Total questions: {len(questions)}")
    print(f"  Successful responses: {sum(1 for r in results if 'error' not in r)}")
    print(f"  Errors: {sum(1 for r in results if 'error' in r)}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation for a model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'swiss-ai/Apertus-8B-Instruct-2509', 'gpt-4', 'claude-sonnet-4.5')",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["cscs", "anthropic", "openai", "google", "ollama"],
        help="API provider (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: results/baseline_evaluation/{model_name}_responses.json)",
    )

    args = parser.parse_args()
    run_evaluation(args.model, args.api, args.output)


if __name__ == "__main__":
    main()

