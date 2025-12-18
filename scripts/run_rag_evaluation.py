#!/usr/bin/env python3
"""
Run RAG evaluation for models on the test set.

Uses the existing RAG pipeline from src/warc_tools/rag/

Usage:
    python scripts/run_rag_evaluation.py --model <model_name> [--test_set <test_set.json>] [--output <output_dir>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from warc_tools.rag.rag_pipeline import RAGConfig, run_rag_query
from warc_tools.indexer.utils import setup_logging
from evaluation_utils import normalize_url

load_dotenv()


def extract_retrieved_docs(nodes) -> List[Dict[str, Any]]:
    """Extract document information from retrieved nodes.
    
    Args:
        nodes: List of NodeWithScore objects from RAG retrieval
        
    Returns:
        List of dictionaries with url, normalized_url, score, and text_preview
    """
    retrieved_docs = []
    retrieved_urls = []
    
    for node in nodes:
        meta = getattr(node.node, "metadata", None) or {}
        url = meta.get("url", "")
        score = float(node.score) if hasattr(node, "score") else 0.0
        text = str(node.node.text)[:200] if hasattr(node.node, "text") else ""
        
        if url:
            normalized_url = normalize_url(url)
            if normalized_url not in retrieved_urls:
                retrieved_docs.append({
                    "url": url,
                    "normalized_url": normalized_url,
                    "score": score,
                    "text_preview": text,
                })
                retrieved_urls.append(normalized_url)
    
    return retrieved_docs


def run_rag_evaluation_for_model(
    model_name: str,
    test_set_path: Path,
    output_dir: Path,
    config: RAGConfig,
    logger,
) -> None:
    """Run RAG evaluation for one model on the test set.
    
    Args:
        model_name: Model identifier
        test_set_path: Path to test set JSON file
        output_dir: Directory to save results
        config: RAG configuration
        logger: Logger instance
    """
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    logger.info(f"Loaded {len(test_set)} questions from {test_set_path}")
    
    # Set LLM model via environment variable
    # The RAG pipeline uses get_llm_from_env() which reads LLM_MODEL
    original_llm = os.environ.get("LLM_MODEL")
    os.environ["LLM_MODEL"] = model_name
    logger.info(f"Using LLM model: {model_name}")
    
    results = []
    errors = []
    
    try:
        for i, item in enumerate(test_set, 1):
            question = item["question"]
            question_id = item["question_id"]
            language = item.get("language", "unknown")
            ground_truth = item.get("ground_truth", "")
            
            logger.info(f"[{i}/{len(test_set)}] Processing Q{question_id}: {question[:60]}...")
            
            try:
                # Run RAG query
                rag_result = run_rag_query(config, question, logger)
                
                # Extract retrieved documents
                retrieved_docs = extract_retrieved_docs(rag_result.nodes)
                retrieved_urls = [doc["url"] for doc in retrieved_docs]
                
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "language": language,
                    "ground_truth": ground_truth,
                    "model_response": rag_result.answer,
                    "retrieved_doc_urls": retrieved_urls,
                    "retrieved_doc_count": len(retrieved_docs),
                })
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing question {question_id}: {e}")
                errors.append({
                    "question_id": question_id,
                    "error": str(e),
                })
                # Continue with next question
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "language": language,
                    "ground_truth": ground_truth,
                    "model_response": "",
                    "retrieved_doc_urls": [],
                    "retrieved_doc_count": 0,
                    "error": str(e),
                })
            
            # Save progress every 10 questions
            if i % 10 == 0:
                model_safe = model_name.replace("/", "_")
                output_file = output_dir / f"{model_safe}_rag_responses.json"
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved ({i}/{len(test_set)} questions)")
    
    finally:
        # Restore original LLM setting
        if original_llm:
            os.environ["LLM_MODEL"] = original_llm
        elif "LLM_MODEL" in os.environ:
            del os.environ["LLM_MODEL"]
    
    # Save final results
    model_safe = model_name.replace("/", "_")
    output_file = output_dir / f"{model_safe}_rag_responses.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Evaluation complete!")
    logger.info(f"  Total questions: {len(test_set)}")
    logger.info(f"  Successful: {len(results) - len(errors)}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"  Results saved to: {output_file}")
    
    if errors:
        error_file = output_dir / f"{model_safe}_rag_errors.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        logger.warning(f"  Errors saved to: {error_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation for a model on the test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., swiss-ai/Apertus-8B-Instruct-2509)",
    )
    parser.add_argument(
        "--test_set",
        type=Path,
        default=Path("test_set/eth_questions_100.json"),
        help="Path to test set JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rag_evaluation"),
        help="Output directory for RAG responses",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logging(log_level, None)
    
    # Check test set exists
    if not args.test_set.exists():
        logger.error(f"Test set not found: {args.test_set}")
        sys.exit(1)
    
    # Get Elasticsearch config from environment
    es_url = os.getenv("ES_URL")
    if not es_url:
        logger.error("ES_URL environment variable not set")
        sys.exit(1)
    
    es_user = os.getenv("ES_USER") or None
    es_password = os.getenv("ES_PASSWORD") or None
    index_name = os.getenv("ES_INDEX_NAME")
    if not index_name:
        logger.error("ES_INDEX_NAME environment variable not set")
        sys.exit(1)
    
    # Create RAG config
    config = RAGConfig(
        es_url=es_url,
        es_user=es_user,
        es_password=es_password,
        index_name=index_name,
        top_k=args.top_k,
    )
    
    logger.info("=" * 60)
    logger.info(f"RAG Evaluation Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Test Set: {args.test_set}")
    logger.info(f"Output: {args.output}")
    logger.info(f"ES Index: {index_name}")
    logger.info(f"Top K: {args.top_k}")
    logger.info("=" * 60)
    
    # Run evaluation
    run_rag_evaluation_for_model(
        args.model,
        args.test_set,
        args.output,
        config,
        logger,
    )


if __name__ == "__main__":
    main()

