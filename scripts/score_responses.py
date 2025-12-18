"""Strict ETH-specific scoring using LLM-as-Judge with ETH-specific criteria."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

load_dotenv()

# Judge settings
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500


def load_judge_prompt() -> str:
    """Load the judge prompt template (lenient for baseline evaluation)."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt_baseline.txt"
    if not prompt_path.exists():
        # Fallback to strict if lenient doesn't exist
        prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt_strict.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_responses(json_path: Path) -> List[Dict[str, Any]]:
    """Load model responses from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    # Strip thinking blocks from responses if present
    for response in responses:
        model_response = response.get("model_response", "")
        if model_response:
            # Remove thinking/reasoning blocks (common in thinking models)
            # Pattern: <think>...</think> or <reasoning>...</reasoning> or similar
            model_response = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL | re.IGNORECASE)
            model_response = re.sub(r'<reasoning>.*?</reasoning>', '', model_response, flags=re.DOTALL | re.IGNORECASE)
            model_response = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL | re.IGNORECASE)
            # Clean up extra whitespace
            model_response = re.sub(r'\n\s*\n', '\n\n', model_response).strip()
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
    """Call the judge LLM to score a response."""
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

        # Handle thinking models (like Kimi-K2-Thinking) that use reasoning_content
        message = resp.choices[0].message
        
        # Check for reasoning_content first (thinking models)
        response_text = None
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            response_text = message.reasoning_content.strip()
        elif hasattr(message, 'content') and message.content:
            response_text = message.content.strip()
        
        if not response_text:
            # Try to get any available text
            response_text = str(message) if message else ""
            if not response_text:
                raise ValueError("No content or reasoning_content in judge response. Message object: " + str(type(message)))

        # Try to extract JSON - multiple strategies
        scores = None
        
        # Strategy 1: Look for JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON object in the text
        if scores is None:
            json_match = re.search(r"\{[^{}]*\"correctness\"[^{}]*\"completeness\"[^{}]*\"result_tag\"[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    scores = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Try to extract scores from reasoning text (for thinking models)
        if scores is None:
            # Extract correctness - try multiple patterns (0-2 integer scale)
            correctness_match = None
            patterns_correctness = [
                r"1\.\s*FACTUAL\s+CORRECTNESS.*?[Ss]core[:\s]+(\d)",
                r"FACTUAL\s+CORRECTNESS.*?[Ss]core[:\s]+(\d)",
                r"FACTUAL\s+CORRECTNESS.*?[Ss]core\s+(?:must\s+be|is)[:\s]+(\d)",
                r'"correctness"\s*:\s*(\d)',
                r'correctness["\']?\s*[:=]\s*(\d)',
                # Look for "Score: 0" or "Score: 2" pattern (most common in judge responses)
                r"(?:correctness|CORRECTNESS|Factual\s+Correctness).*?[Ss]core[:\s]+(\d)",
            ]
            
            for pattern in patterns_correctness:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        val = int(match.group(1))
                        if 0 <= val <= 2:
                            correctness_match = val
                            break
                    except (ValueError, IndexError):
                        continue
            
            # Extract completeness - try multiple patterns (0-2 integer scale)
            completeness_match = None
            patterns_completeness = [
                r"2\.\s*COMPLETENESS.*?[Ss]core[:\s]+(\d)",
                r"COMPLETENESS.*?[Ss]core[:\s]+(\d)",
                r"COMPLETENESS.*?[Ss]core\s+(?:must\s+be|is)[:\s]+(\d)",
                r'"completeness"\s*:\s*(\d)',
                r'completeness["\']?\s*[:=]\s*(\d)',
                # Look for "Score: 0" or "Score: 2" pattern (most common in judge responses)
                r"(?:completeness|COMPLETENESS|Completeness).*?[Ss]core[:\s]+(\d)",
            ]
            
            for pattern in patterns_completeness:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        val = int(match.group(1))
                        if 0 <= val <= 2:
                            completeness_match = val
                            break
                    except (ValueError, IndexError):
                        continue
            
            # Extract result_tag - try multiple patterns
            result_tag = None
            tag_patterns = [
                r'"result_tag"\s*:\s*["\']?(Correct|Generic|Refusal|Hallucination)["\']?',
                r"result_tag[:\s]+[\"']?(Correct|Generic|Refusal|Hallucination)[\"']?",
                r"tag[:\s]+[\"']?(Correct|Generic|Refusal|Hallucination)[\"']?",
                r'result_tag["\']?\s*[:=]\s*["\']?(Correct|Generic|Refusal|Hallucination)["\']?',
            ]
            for pattern in tag_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    result_tag = match.group(1).capitalize()
                    break
            
            # Infer tag from reasoning if not found
            if result_tag is None:
                text_lower = response_text.lower()
                if "hallucination" in text_lower or "invents" in text_lower or "invented" in text_lower:
                    result_tag = "Hallucination"
                elif any(phrase in text_lower for phrase in ["refusal", "don't know", "cannot answer", "i don't know", "i cannot"]):
                    result_tag = "Refusal"
                elif correctness_match == 2 and completeness_match == 2:
                    result_tag = "Correct"
                elif "generic" in text_lower or "general" in text_lower or "general advice" in text_lower:
                    result_tag = "Generic"
                else:
                    # Default based on scores
                    if correctness_match == 0 and completeness_match == 0:
                        result_tag = "Generic"  # Default for 0/0
                    else:
                        result_tag = "Generic"  # Default
            
            # If we found both scores, create scores dict
            if correctness_match is not None and completeness_match is not None:
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag,
                    "reasoning": response_text,  # Keep full reasoning, no truncation
                }
            # If we only found one, try to infer the other or use defaults
            elif correctness_match is not None:
                # Found correctness but not completeness - infer completeness
                if correctness_match == 0:
                    completeness_match = 0
                else:
                    completeness_match = max(0, correctness_match - 1)  # Slightly lower for partial
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag if result_tag else "Generic",
                    "reasoning": response_text,  # Keep full reasoning, no truncation
                }
            elif completeness_match is not None:
                # Found completeness but not correctness - infer correctness
                if completeness_match == 0:
                    correctness_match = 0
                else:
                    correctness_match = max(0, completeness_match - 1)  # Slightly lower for partial
                scores = {
                    "correctness": correctness_match,
                    "completeness": completeness_match,
                    "result_tag": result_tag if result_tag else "Generic",
                    "reasoning": response_text,  # Keep full reasoning, no truncation
                }
        
        # Strategy 4: Try to parse entire response as JSON
        if scores is None:
            try:
                scores = json.loads(response_text)
            except json.JSONDecodeError:
                pass
        
        # If still no scores after all attempts, try to infer from text analysis
        if scores is None:
            # Last resort: analyze the reasoning text to infer scores
            text_lower = response_text.lower()
            
            # Check if it mentions ETH tools
            eth_tools = ['ethis', 'asvz', 'moodle', 'card-ethz', 'web print', 'staffnet', 'euler']
            mentions_eth_tool = any(tool in text_lower for tool in eth_tools)
            
            # Check if it's generic advice
            generic_phrases = ['generic', 'general advice', 'does not mention', 'misses', 'completely generic']
            is_generic = any(phrase in text_lower for phrase in generic_phrases)
            
            # Check if it's refusal
            refusal_phrases = ["don't know", "cannot", "i don't", "refusal"]
            is_refusal = any(phrase in text_lower for phrase in refusal_phrases)
            
            # Check if it's hallucination
            hallucination_phrases = ['invents', 'invented', 'hallucination', 'fake', 'does not exist']
            is_hallucination = any(phrase in text_lower for phrase in hallucination_phrases)
            
            # Infer scores based on analysis (0-2 scale)
            if mentions_eth_tool and not is_generic:
                correctness = 2
                completeness = 2
                result_tag = "Correct"
            elif is_refusal:
                correctness = 0
                completeness = 0
                result_tag = "Refusal"
            elif is_hallucination:
                correctness = 0
                completeness = 0
                result_tag = "Hallucination"
            elif is_generic or not mentions_eth_tool:
                correctness = 0
                completeness = 0
                result_tag = "Generic"
            else:
                # Default fallback
                correctness = 0
                completeness = 0
                result_tag = "Generic"
            
                scores = {
                    "correctness": correctness,
                    "completeness": completeness,
                    "result_tag": result_tag,
                    "reasoning": response_text,  # Keep full reasoning, no truncation
                }

        # If still no scores, use fallback
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
        result_tag = str(scores.get("result_tag", "Generic"))
        reasoning = str(scores.get("reasoning", ""))

        # Clamp scores to valid range (0-2)
        correctness = max(0, min(2, correctness))
        completeness = max(0, min(2, completeness))
        
        # More lenient logic: Only enforce 0/0 for truly generic or refusal
        # Partial answers get at least 1 point if they provide any ETH-relevant info
        if result_tag == "Partial":
            # Partial answers should get at least 1 point if judge gave them
            correctness = max(1, correctness) if correctness > 0 else correctness
            completeness = max(1, completeness) if completeness > 0 else completeness
        elif result_tag in ["Generic", "Refusal"]:
            # Only enforce 0/0 for truly generic or refusal
            correctness = 0
            completeness = 0

        # Validate result_tag (include Partial for lenient scoring)
        valid_tags = ["Correct", "Partial", "Generic", "Refusal", "Hallucination"]
        if result_tag not in valid_tags:
            result_tag = "Generic"
        
        # STRICT AUDITOR ENFORCEMENT: Generic or Refusal = correctness must be 0
        # We are testing for internal institutional knowledge; "true but general" info is a failure
        if result_tag in ["Generic", "Refusal"]:
            correctness = 0
            completeness = 0

        return {
            "correctness": correctness,
            "completeness": completeness,
            "result_tag": result_tag,
            "reasoning": reasoning,  # Full reasoning, no truncation
            "judge_model": judge_model,
        }

    except Exception as e:
        print(f"Error calling judge: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Judge API call failed: {str(e)}")


def score_responses(
    model_name: str,
    judge_model: str = "moonshotai/Kimi-K2-Thinking",
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Score model responses using strict ETH-specific LLM-as-Judge."""
    # Determine input file
    if input_file is None:
        results_dir = Path(__file__).parent.parent / "results" / "baseline_evaluation"
        model_safe = model_name.replace("/", "_").replace(" ", "_")
        input_file = results_dir / f"{model_safe}_responses.json"

    if not input_file.exists():
        print(f"Error: Responses file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Load responses
    responses = load_responses(input_file)
    print(f"Loaded {len(responses)} responses from {input_file}")

    # Load judge prompt
    prompt_template = load_judge_prompt()

    # Initialize judge client
    # Check if judge is a CSCS model or OpenAI model
    is_cscs_judge = "/" in judge_model or judge_model not in ["gpt-4o", "gpt-4-turbo"]
    
    if is_cscs_judge:
        # Use CSCS API for judge
        judge_api_key = os.getenv("CSCS_API_KEY") or os.getenv("LLM_API_KEY")
        judge_base_url = os.getenv("CSCS_BASE_URL", "https://api.swissai.cscs.ch/v1")
        if not judge_api_key:
            print(
                "Error: CSCS_API_KEY or LLM_API_KEY environment variable required for CSCS judge",
                file=sys.stderr,
            )
            sys.exit(1)
        judge_client = openai.Client(api_key=judge_api_key, base_url=judge_base_url)
        print(f"Using CSCS API for judge: {judge_model}")
    else:
        # Use OpenAI API for judge
        judge_api_key = os.getenv("OPENAI_API_KEY")
        if not judge_api_key:
            print(
                "Error: OPENAI_API_KEY environment variable required for OpenAI judge",
                file=sys.stderr,
            )
            sys.exit(1)
        judge_client = openai.Client(api_key=judge_api_key)
        print(f"Using OpenAI API for judge: {judge_model}")

    # Check for existing scores
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

    # Score each response
    scores = []
    skipped = 0

    print(f"\nScoring responses using strict ETH-specific judge: {judge_model}")
    print(f"Total questions: {len(responses)}\n")

    for i, response in enumerate(responses, 1):
        q_id = response["question_id"]

        # Skip if already scored
        if q_id in existing_scores:
            scores.append(existing_scores[q_id])
            skipped += 1
            continue

        question = response["question"]
        ground_truth = response.get("ground_truth", "")
        model_response = response.get("model_response", "")

        print(f"[{i}/{len(responses)}] Scoring question {q_id}...", end=" ", flush=True)

        try:
            judge_scores = call_judge(
                judge_client,
                judge_model,
                question,
                ground_truth,
                model_response,
                prompt_template,
            )

            # Calculate aggregate score (0-1 range from 0-2 scale)
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

        except KeyboardInterrupt:
            print("\n\nScoring interrupted. Saving partial results...")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            # Retry once with delay
            try:
                print(f"Retrying question {q_id}...")
                time.sleep(3)  # Wait before retry
                judge_scores = call_judge(
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

        # Rate limiting
        time.sleep(0.5)  # Small delay to avoid rate limits

        # Save progress every 10 questions
        if i % 10 == 0:
            all_scores = existing_scores.copy()
            for score in scores:
                all_scores[score["question_id"]] = score

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(all_scores.values()), f, indent=2, ensure_ascii=False)
            print(f"\nProgress saved ({i}/{len(responses)} questions)")

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
