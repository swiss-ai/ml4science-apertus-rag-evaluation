"""Score RAG responses using LLM-as-Judge with retrieval quality metrics."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation_utils import (
    call_judge,
    load_judge_prompt,
    load_responses_with_cleaning,
    normalize_url,
    save_scores_progress,
    setup_judge_client,
)


def normalize_url_for_retrieval(url: str) -> str:
    """Normalize URL for retrieval metrics comparison."""
    return normalize_url(url)


def calculate_retrieval_metrics(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_1: Optional[str],
    relevant_doc_2: Optional[str],
) -> Dict[str, Any]:
    """Calculate retrieval quality metrics (precision, recall, matches).
    
    Args:
        retrieved_docs: List of retrieved document dictionaries
        relevant_doc_1: First ground truth relevant document URL
        relevant_doc_2: Second ground truth relevant document URL
        
    Returns:
        Dictionary with precision, recall, found flags, and counts
    """
    if not retrieved_docs:
        return {
            "precision": None,
            "recall": None,
            "found_relevant_doc_1": False,
            "found_relevant_doc_2": False,
            "num_relevant_found": 0,
            "num_ground_truth_relevant": 0,
            "retrieved_doc_count": 0,
        }
    
    # Normalize ground truth URLs
    ground_truth_urls = []
    if relevant_doc_1:
        ground_truth_urls.append(normalize_url_for_retrieval(relevant_doc_1))
    if relevant_doc_2:
        ground_truth_urls.append(normalize_url_for_retrieval(relevant_doc_2))
    
    # Extract retrieved URLs
    retrieved_urls = []
    for doc in retrieved_docs:
        url = doc.get("url", "") or doc.get("normalized_url", "")
        if url:
            retrieved_urls.append(normalize_url_for_retrieval(url))
    
    # Check which relevant docs were found
    found_relevant_1 = False
    found_relevant_2 = False
    
    if relevant_doc_1:
        normalized_gt1 = normalize_url_for_retrieval(relevant_doc_1)
        found_relevant_1 = any(
            normalized_gt1 in ret_url or ret_url in normalized_gt1
            for ret_url in retrieved_urls
        )
    
    if relevant_doc_2:
        normalized_gt2 = normalize_url_for_retrieval(relevant_doc_2)
        found_relevant_2 = any(
            normalized_gt2 in ret_url or ret_url in normalized_gt2
            for ret_url in retrieved_urls
        )
    
    num_relevant_found = sum([found_relevant_1, found_relevant_2])
    num_ground_truth_relevant = len(ground_truth_urls)
    
    # Calculate precision and recall
    precision = (
        num_relevant_found / len(retrieved_urls) if retrieved_urls else 0.0
    )
    recall = (
        num_relevant_found / num_ground_truth_relevant
        if num_ground_truth_relevant > 0
        else None
    )
    
    return {
        "precision": precision if precision > 0 else 0.0,
        "recall": recall if recall is not None else 0.0,
        "found_relevant_doc_1": found_relevant_1,
        "found_relevant_doc_2": found_relevant_2,
        "num_relevant_found": num_relevant_found,
        "num_ground_truth_relevant": num_ground_truth_relevant,
        "retrieved_doc_count": len(retrieved_urls),
    }


def auto_suggest_rag_tag(
    retrieval_metrics: Dict[str, Any],
    correctness: int,
    completeness: int,
    result_tag: str,
) -> str:
    """Suggest RAG-specific tag based on retrieval metrics and scores.
    
    Args:
        retrieval_metrics: Dictionary with retrieval quality metrics
        correctness: Correctness score (0-2)
        completeness: Completeness score (0-2)
        result_tag: Base result tag from judge
        
    Returns:
        RAG-specific tag (Correct, Partial, retrieval_failure, ignored_context, etc.)
    """
    recall = retrieval_metrics.get("recall", 0)
    if recall is None:
        recall = 0.0
    num_found = retrieval_metrics.get("num_relevant_found", 0)
    
    # If answered correctly or partially - merge into Correct/Partial
    if correctness >= 1.5 and completeness >= 1.5:
        # If the answer is correct, it's a success regardless of retrieval
        return "Correct"
    elif correctness >= 1 or completeness >= 1:
        # Partial answers are also valuable - mark as Partial
        return "Partial"
    
    # If answered incorrectly
    if correctness == 0:
        if recall == 0 or recall is None:
            return "retrieval_failure"  # Didn't find the right docs
        elif recall >= 0.5:
            return "ignored_context"  # Had the right docs but didn't use them
        else:
            return "partial_retrieval"  # Found some but not all needed docs
    
    # Use baseline tag if no RAG-specific tag applies
    return result_tag


def load_rag_responses(json_path: Path) -> List[Dict[str, Any]]:
    """Load RAG responses from JSON file and clean thinking blocks."""
    return load_responses_with_cleaning(json_path)


def score_rag_responses(
    model_name: str,
    judge_model: str,
    responses_file: Path,
    test_set_file: Path,
    output_file: Path,
) -> None:
    """Score RAG responses using LLM-as-Judge with retrieval metrics.
    
    Args:
        model_name: Model identifier
        judge_model: Judge model identifier
        responses_file: Path to RAG responses JSON file
        test_set_file: Path to test set JSON file with ground truth
        output_file: Path to output scores JSON file
    """
    # Load data
    responses = load_rag_responses(responses_file)
    with open(test_set_file, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    # Create map for ground truth
    ground_truth_map = {item["question_id"]: item for item in test_set}
    
    judge_prompt = load_judge_prompt("rag")
    
    try:
        client = setup_judge_client(judge_model)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Load existing scores if resuming
    existing_scores = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_scores_list = json.load(f)
            existing_scores = {s["question_id"]: s for s in existing_scores_list}
        print(f"Loaded {len(existing_scores)} existing scores")
    
    scores = []
    skipped = 0
    
    for i, response in enumerate(responses, 1):
        q_id = response["question_id"]
        
        # Skip if already scored
        if q_id in existing_scores:
            skipped += 1
            continue
        
        ground_truth_item = ground_truth_map.get(q_id)
        if not ground_truth_item:
            print(f"WARNING: No ground truth for question {q_id}")
            continue
        
        question = response["question"]
        model_response = response.get("model_response", "")
        ground_truth = ground_truth_item.get("ground_truth", "")
        
        print(f"[{i}/{len(responses)}] Scoring Q{q_id}...")
        
        judge_result = call_judge(
            client,
            judge_model,
            question,
            ground_truth,
            model_response,
            judge_prompt,
        )
        
        # Calculate retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(
            response.get("retrieved_docs", []),
            ground_truth_item.get("relevant_doc_1"),
            ground_truth_item.get("relevant_doc_2"),
        )
        
        # Auto-suggest RAG tag
        rag_tag = auto_suggest_rag_tag(
            retrieval_metrics,
            judge_result["correctness"],
            judge_result["completeness"],
            judge_result["result_tag"],
        )
        
        # Calculate aggregate score
        aggregate_score = (
            judge_result["correctness"] + judge_result["completeness"]
        ) / 4.0
        
        score_entry = {
            "question_id": q_id,
            "correctness": judge_result["correctness"],
            "completeness": judge_result["completeness"],
            "aggregate_score": aggregate_score,
            "result_tag": rag_tag,
            "reasoning": judge_result["reasoning"],
            "retrieval_metrics": retrieval_metrics,
        }
        
        scores.append(score_entry)
        
        time.sleep(0.5)  # Rate limiting
        
        if i % 10 == 0:
            save_scores_progress(output_file, existing_scores, scores, i, len(responses))
    
    # Final save
    all_scores = existing_scores.copy()
    for score in scores:
        all_scores[score["question_id"]] = score
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(all_scores.values()), f, indent=2, ensure_ascii=False)
    
    print(f"\nScoring complete!")
    print(f"  Total scored: {len(scores)}")
    print(f"  Skipped (already scored): {skipped}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score RAG responses using strict ETH-specific LLM-as-Judge"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., swiss-ai/Apertus-8B-Instruct-2509)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="moonshotai/Kimi-K2-Thinking",
        help="Judge model name (default: moonshotai/Kimi-K2-Thinking)",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        help="Input RAG responses JSON file (default: results/rag_evaluation/{model}_rag_responses.json)",
    )
    parser.add_argument(
        "--test_set",
        type=Path,
        default=Path("test_set/eth_questions_100.json"),
        help="Test set JSON file with ground truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output scores JSON file (default: results/rag_evaluation/{model}_rag_scores.json)",
    )

    args = parser.parse_args()

    # Determine file paths
    model_safe = args.model.replace("/", "_")
    if args.responses is None:
        args.responses = Path("results/rag_evaluation") / f"{model_safe}_rag_responses.json"
    if args.output is None:
        args.output = Path("results/rag_evaluation") / f"{model_safe}_rag_scores.json"
    elif args.output.is_dir() or (not args.output.suffix and not args.output.exists()):
        # If output is a directory or has no extension, append the filename
        args.output = args.output / f"{model_safe}_rag_scores.json"

    if not args.responses.exists():
        print(f"ERROR: Responses file not found: {args.responses}")
        sys.exit(1)

    if not args.test_set.exists():
        print(f"ERROR: Test set file not found: {args.test_set}")
        sys.exit(1)

    score_rag_responses(
        args.model,
        args.judge,
        args.responses,
        args.test_set,
        args.output,
    )

