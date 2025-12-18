"""Score RAG responses using LLM-as-Judge with retrieval quality metrics."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv

load_dotenv()

# Judge settings
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    if not url:
        return ""
    url = url.strip().rstrip("/")
    # Remove protocol for comparison
    parsed = urlparse(url)
    normalized = f"{parsed.netloc}{parsed.path}".rstrip("/")
    return normalized.lower()


def calculate_retrieval_metrics(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_1: Optional[str],
    relevant_doc_2: Optional[str],
) -> Dict[str, Any]:
    """Calculate retrieval quality metrics."""
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
        ground_truth_urls.append(normalize_url(relevant_doc_1))
    if relevant_doc_2:
        ground_truth_urls.append(normalize_url(relevant_doc_2))
    
    # Extract retrieved URLs
    retrieved_urls = []
    for doc in retrieved_docs:
        url = doc.get("url", "") or doc.get("normalized_url", "")
        if url:
            retrieved_urls.append(normalize_url(url))
    
    # Check which relevant docs were found
    found_relevant_1 = False
    found_relevant_2 = False
    
    if relevant_doc_1:
        normalized_gt1 = normalize_url(relevant_doc_1)
        found_relevant_1 = any(
            normalized_gt1 in ret_url or ret_url in normalized_gt1
            for ret_url in retrieved_urls
        )
    
    if relevant_doc_2:
        normalized_gt2 = normalize_url(relevant_doc_2)
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
    """Suggest RAG-specific tag based on retrieval metrics and scores."""
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


def load_judge_prompt() -> str:
    """Load the judge prompt template (favorable for RAG evaluation)."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt_rag.txt"
    if not prompt_path.exists():
        # Fallback to strict if favorable doesn't exist
        prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt_strict.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_rag_responses(json_path: Path) -> List[Dict[str, Any]]:
    """Load RAG responses from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    # Strip thinking blocks from responses if present
    for response in responses:
        model_response = response.get("model_response", "")
        if model_response:
            model_response = re.sub(
                r"<think>.*?</think>",
                "",
                model_response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            model_response = re.sub(
                r"<reasoning>.*?</reasoning>",
                "",
                model_response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            model_response = re.sub(r"\n\s*\n", "\n\n", model_response).strip()
            response["model_response"] = model_response
    
    return responses


def call_judge(
    client: openai.Client,
    judge_model: str,
    question: str,
    ground_truth: str,
    model_response: str,
    prompt_template: str,
) -> Dict[str, Any]:
    """Call the judge LLM to score a response (same logic as baseline)."""
    # Format the prompt
    prompt = prompt_template.format(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a Strict Auditor for ETH Zurich. Respond only with valid JSON, no additional text.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )

        # Handle thinking models
        message = resp.choices[0].message
        response_text = None
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            response_text = message.reasoning_content.strip()
        elif hasattr(message, "content") and message.content:
            response_text = message.content.strip()

        if not response_text:
            response_text = str(message) if message else ""
            if not response_text:
                raise ValueError(
                    "No content or reasoning_content in judge response."
                )

        # Extract scores (same logic as baseline score_responses.py)
        scores = None

        # Strategy 1: Try to find JSON block
        json_match = re.search(r"\{[^{}]*\"correctness\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 2: Extract individual fields with regex
        if scores is None:
            correctness_match = None
            completeness_match = None
            result_tag = None

            # Try to find correctness
            corr_match = re.search(
                r'"correctness"\s*:\s*(\d+)', response_text, re.IGNORECASE
            )
            if corr_match:
                correctness_match = int(corr_match.group(1))

            # Try to find completeness
            comp_match = re.search(
                r'"completeness"\s*:\s*(\d+)', response_text, re.IGNORECASE
            )
            if comp_match:
                completeness_match = int(comp_match.group(1))

            # Try to find result_tag
            tag_match = re.search(
                r'"result_tag"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE
            )
            if tag_match:
                result_tag = tag_match.group(1)

            if correctness_match is not None and completeness_match is not None:
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag if result_tag else "Generic",
                    "reasoning": response_text,
                }
            elif correctness_match is not None:
                completeness_match = max(0, correctness_match - 1) if correctness_match > 0 else 0
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag if result_tag else "Generic",
                    "reasoning": response_text,
                }
            elif completeness_match is not None:
                correctness_match = max(0, completeness_match - 1) if completeness_match > 0 else 0
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag if result_tag else "Generic",
                    "reasoning": response_text,
                }

        # Strategy 3: Try to parse entire response as JSON
        if scores is None:
            try:
                scores = json.loads(response_text)
            except json.JSONDecodeError:
                pass

        # Fallback: use defaults
        if scores is None:
            scores = {
                "correctness": 0,
                "completeness": 0,
                "result_tag": "Generic",
                "reasoning": response_text,
            }

        # Validate and normalize scores (0-2 integer scale)
        correctness = int(scores.get("correctness", 0))
        completeness = int(scores.get("completeness", 0))
        result_tag = scores.get("result_tag", "Generic")

        # More lenient logic for RAG: Only enforce 0/0 for truly generic or refusal
        # Partial answers get at least 1 point if they provide any ETH-relevant info
        if result_tag == "Partial":
            # Partial answers should get at least 1 point if judge gave them
            correctness = max(1, correctness) if correctness > 0 else correctness
            completeness = max(1, completeness) if completeness > 0 else completeness
        elif result_tag in ["Generic", "Refusal"]:
            # Only enforce 0/0 for truly generic or refusal
            correctness = 0
            completeness = 0

        # Clamp to valid range
        correctness = max(0, min(2, correctness))
        completeness = max(0, min(2, completeness))

        return {
            "correctness": correctness,
            "completeness": completeness,
            "result_tag": result_tag,
            "reasoning": scores.get("reasoning", response_text),
        }

    except Exception as e:
        print(f"ERROR: Judge call failed: {e}")
        return {
            "correctness": 0,
            "completeness": 0,
            "result_tag": "Generic",
            "reasoning": f"Judge error: {str(e)}",
        }


def score_rag_responses(
    model_name: str,
    judge_model: str,
    responses_file: Path,
    test_set_file: Path,
    output_file: Path,
) -> None:
    """Score RAG responses using LLM-as-Judge."""
    # Load data
    responses = load_rag_responses(responses_file)
    with open(test_set_file, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    # Create map for ground truth
    ground_truth_map = {item["question_id"]: item for item in test_set}
    
    # Load judge prompt
    judge_prompt = load_judge_prompt()
    
    # Setup judge client (same logic as baseline score_responses.py)
    judge_base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("CSCS_BASE_URL") or "https://api.swissai.cscs.ch/v1"
    judge_api_key = os.getenv("JUDGE_API_KEY") or os.getenv("CSCS_API_KEY") or os.getenv("LLM_API_KEY")
    
    if not judge_api_key:
        raise ValueError(
            "JUDGE_API_KEY, CSCS_API_KEY, or LLM_API_KEY must be set"
        )
    
    client = openai.Client(base_url=judge_base_url, api_key=judge_api_key)
    
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
        
        # Call judge
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
        
        # Rate limiting
        time.sleep(0.5)
        
        # Save progress every 10 questions
        if i % 10 == 0:
            all_scores = existing_scores.copy()
            for score in scores:
                all_scores[score["question_id"]] = score
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(all_scores.values()), f, indent=2, ensure_ascii=False)
            print(f"Progress saved ({i}/{len(responses)} questions)")
    
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

