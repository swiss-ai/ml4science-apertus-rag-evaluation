"""Strict ETH-specific scoring using LLM-as-Judge with ETH-specific criteria."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from evaluation_utils import (
    call_judge,
    load_judge_prompt,
    load_responses_with_cleaning,
    save_scores_progress,
    setup_judge_client,
)


def load_responses(json_path: Path) -> List[Dict]:
    """Load model responses from JSON file and clean thinking blocks.
    
    Args:
        json_path: Path to responses JSON file
        
    Returns:
        List of response dictionaries with cleaned model_response text
    """
    return load_responses_with_cleaning(json_path)


def call_judge_wrapper(
    client,
    judge_model: str,
    question: str,
    ground_truth: str,
    model_response: str,
    prompt_template: str,
) -> Dict:
    """Wrapper for call_judge from evaluation_utils."""
    return call_judge(client, judge_model, question, ground_truth, model_response, prompt_template)


def score_responses(
    model_name: str,
    judge_model: str = "moonshotai/Kimi-K2-Thinking",
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Score model responses using LLM-as-Judge.
    
    Args:
        model_name: Model identifier
        judge_model: Judge model identifier (default: moonshotai/Kimi-K2-Thinking)
        input_file: Optional input responses file (defaults to results/baseline_evaluation/)
        output_file: Optional output scores file (defaults to results/baseline_evaluation/)
    """
    if input_file is None:
        results_dir = Path(__file__).parent.parent / "results" / "baseline_evaluation"
        model_safe = model_name.replace("/", "_").replace(" ", "_")
        input_file = results_dir / f"{model_safe}_responses.json"

    if not input_file.exists():
        print(f"Error: Responses file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    responses = load_responses(input_file)
    print(f"Loaded {len(responses)} responses from {input_file}")

    prompt_template = load_judge_prompt("baseline")
    
    try:
        judge_client = setup_judge_client(judge_model)
        print(f"Using judge: {judge_model}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if output_file is None:
        results_dir = Path(__file__).parent.parent / "results" / "baseline_evaluation"
        model_safe = model_name.replace("/", "_").replace(" ", "_")
        output_file = results_dir / f"{model_safe}_scores.json"

    existing_scores = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_scores = {s["question_id"]: s for s in existing_data}
            else:
                existing_scores = {}
        print(f"Found {len(existing_scores)} existing scores. Will skip those questions.")

    scores = []
    skipped = 0

    print(f"\nScoring responses using strict ETH-specific judge: {judge_model}")
    print(f"Total questions: {len(responses)}\n")

    for i, response in enumerate(responses, 1):
        q_id = response["question_id"]

        if q_id in existing_scores:
            scores.append(existing_scores[q_id])
            skipped += 1
            continue

        question = response["question"]
        ground_truth = response.get("ground_truth", "")
        model_response = response.get("model_response", "")

        print(f"[{i}/{len(responses)}] Scoring question {q_id}...", end=" ", flush=True)

        try:
            judge_scores = call_judge_wrapper(
                judge_client,
                judge_model,
                question,
                ground_truth,
                model_response,
                prompt_template,
            )

            aggregate_score = (judge_scores["correctness"] + judge_scores["completeness"]) / 4.0

            score_data = {
                "question_id": q_id,
                "correctness": judge_scores["correctness"],
                "completeness": judge_scores["completeness"],
                "aggregate_score": aggregate_score,
                "result_tag": judge_scores["result_tag"],
                "reasoning": judge_scores.get("reasoning", ""),
            }

            scores.append(score_data)
            print(
                f"OK (C={judge_scores['correctness']}, "
                f"Comp={judge_scores['completeness']}, "
                f"Tag={judge_scores['result_tag']}, "
                f"Score={aggregate_score:.2f})"
            )

        except KeyboardInterrupt:
            print("\n\nScoring interrupted. Saving partial results...")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            try:
                print(f"Retrying question {q_id}...")
                time.sleep(3)
                judge_scores = call_judge_wrapper(
                    judge_client,
                    judge_model,
                    question,
                    ground_truth,
                    model_response,
                    prompt_template,
                )
                
                aggregate_score = (
                    judge_scores["correctness"] + judge_scores["completeness"]
                ) / 4.0

                score_data = {
                    "question_id": q_id,
                    "correctness": judge_scores["correctness"],
                    "completeness": judge_scores["completeness"],
                    "aggregate_score": aggregate_score,
                    "result_tag": judge_scores["result_tag"],
                    "reasoning": judge_scores.get("reasoning", ""),
                }

                scores.append(score_data)
                print(
                    f"OK (C={judge_scores['correctness']}, "
                    f"Comp={judge_scores['completeness']}, "
                    f"Tag={judge_scores['result_tag']}, "
                    f"Score={aggregate_score:.2f})"
                )
            except Exception as retry_error:
                print(f"ERROR: Retry failed for question {q_id}: {retry_error}")
                print(f"WARNING: Skipping question {q_id} - persistent error")
                continue

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
    parser = argparse.ArgumentParser(description="Score model responses using strict ETH-specific LLM-as-Judge")
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
        "--input",
        type=Path,
        help="Input responses JSON file (default: results/{model}_responses.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output scores JSON file (default: results/baseline_evaluation/{model}_scores.json)",
    )

    args = parser.parse_args()
    score_responses(args.model, args.judge, args.input, args.output)
