#!/usr/bin/env python3
"""
Comprehensive evaluation script for RAG and baseline answers.

This script:
1. Runs RAG evaluation (if needed) to get answers WITH retrieval
2. Computes evaluation metrics: Exact Match, Multiple Choice, LLM-as-Judge
3. Formats output with requested columns: question, golden answer, qwen answer wo rag, qwen answer with rag
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import re

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenAI for LLM-as-judge
try:
    import openai
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


def require_env(name: str) -> str:
    """Require an environment variable."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


def get_judge_llm_client() -> openai.Client:
    """Get OpenAI-compatible client for LLM-as-judge."""
    # Use same API as main LLM, or specify JUDGE_LLM_* env vars
    api_key = os.getenv("JUDGE_LLM_API_KEY") or require_env("LLM_API_KEY")
    base_url = os.getenv("JUDGE_LLM_BASE_URL") or require_env("LLM_BASE_URL")
    return openai.Client(api_key=api_key, base_url=base_url)


def compute_exact_match(predicted: str, reference: str) -> bool:
    """
    Compute exact match between predicted and reference answers.
    
    Normalizes whitespace and case for comparison.
    """
    if not predicted or not reference:
        return False
    
    # Normalize: lowercase, strip, collapse whitespace
    pred_norm = re.sub(r'\s+', ' ', predicted.lower().strip())
    ref_norm = re.sub(r'\s+', ' ', reference.lower().strip())
    
    return pred_norm == ref_norm


def extract_multiple_choice_answer(text: str) -> Optional[str]:
    """
    Extract multiple choice answer from text (e.g., "A", "B", "C", "D").
    
    Looks for patterns like:
    - "Answer: A"
    - "The answer is B"
    - "(A)", "(B)", etc.
    """
    if not text:
        return None
    
    # Pattern 1: "Answer: A" or "Answer is A"
    match = re.search(r'answer\s*:?\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: "(A)" or "[A]"
    match = re.search(r'[\(\[]([A-E])[\)\]]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Just a single letter at start/end
    match = re.search(r'^\s*([A-E])\s*[\.:]?\s*$', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def compute_multiple_choice_score(predicted: str, reference: str) -> Optional[bool]:
    """
    Compute multiple choice score if both answers are single letters.
    
    Returns True if match, False if mismatch, None if not applicable.
    """
    pred_choice = extract_multiple_choice_answer(predicted)
    ref_choice = extract_multiple_choice_answer(reference)
    
    if pred_choice is None or ref_choice is None:
        return None  # Not a multiple choice question
    
    return pred_choice == ref_choice


def llm_as_judge(
    client: openai.Client,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    model: str = None,
) -> Dict[str, Any]:
    """
    Use an LLM to judge if the predicted answer is correct.
    
    Returns a dict with:
    - score: float (0.0 to 1.0)
    - reasoning: str
    - correct: bool
    """
    if not model:
        model = os.getenv("JUDGE_LLM_MODEL") or "Mistral-7B-Instruct"
    
    prompt = f"""You are an expert evaluator. Judge whether the predicted answer correctly answers the question, compared to the reference answer.

Question: {question}

Reference Answer: {reference_answer}

Predicted Answer: {predicted_answer}

Evaluate the predicted answer on:
1. Correctness: Does it answer the question correctly?
2. Completeness: Does it cover the key points from the reference?
3. Accuracy: Are the facts correct?

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "correct": <true or false>,
    "reasoning": "<brief explanation>"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "score": float(result.get("score", 0.0)),
                "correct": bool(result.get("correct", False)),
                "reasoning": result.get("reasoning", ""),
            }
        else:
            # Fallback: try to parse score from text
            score_match = re.search(r'score["\']?\s*:?\s*([0-9.]+)', content, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            return {
                "score": min(1.0, max(0.0, score)),
                "correct": score >= 0.5,
                "reasoning": content[:200],
            }
    except Exception as e:
        print(f"WARNING: LLM-as-judge failed: {e}", file=sys.stderr)
        return {
            "score": 0.0,
            "correct": False,
            "reasoning": f"Error: {str(e)}",
        }


def llm_compare_answers(
    client: openai.Client,
    question: str,
    reference_answer: str,
    baseline_answer: str,
    rag_answer: str,
    model: str = None,
) -> Dict[str, Any]:
    """
    Use an LLM to compare baseline (no RAG) vs RAG answers and determine which performs better.
    
    Returns a dict with:
    - winner: str ("baseline", "rag", or "tie")
    - baseline_score: float (0.0 to 1.0)
    - rag_score: float (0.0 to 1.0)
    - reasoning: str
    - rag_improved: bool
    """
    if not model:
        model = os.getenv("JUDGE_LLM_MODEL") or "Mistral-7B-Instruct"
    
    prompt = f"""You are an expert evaluator. Compare two answers to determine which one better answers the question.

Question: {question}

Reference Answer (Ground Truth): {reference_answer}

Answer 1 (Baseline - No RAG): {baseline_answer}

Answer 2 (With RAG): {rag_answer}

Evaluate both answers on:
1. Correctness: Does it answer the question correctly?
2. Completeness: Does it cover the key points from the reference?
3. Accuracy: Are the facts correct?
4. Relevance: How well does it address the question?

Compare the two answers and determine:
- Which answer is better overall?
- Did RAG improve the answer quality?

Respond in JSON format:
{{
    "winner": <"baseline", "rag", or "tie">,
    "baseline_score": <float between 0.0 and 1.0>,
    "rag_score": <float between 0.0 and 1.0>,
    "rag_improved": <true or false>,
    "reasoning": "<brief explanation of comparison>"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "score": float(result.get("score", 0.0)),
                "correct": bool(result.get("correct", False)),
                "reasoning": result.get("reasoning", ""),
            }
        else:
            # Fallback: try to parse score from text
            score_match = re.search(r'score["\']?\s*:?\s*([0-9.]+)', content, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            return {
                "score": min(1.0, max(0.0, score)),
                "correct": score >= 0.5,
                "reasoning": content[:200],
            }
    except Exception as e:
        print(f"WARNING: LLM-as-judge failed: {e}", file=sys.stderr)
        return {
            "score": 0.0,
            "correct": False,
            "reasoning": f"Error: {str(e)}",
        }


def llm_compare_answers(
    client: openai.Client,
    question: str,
    reference_answer: str,
    baseline_answer: str,
    rag_answer: str,
    model: str = None,
) -> Dict[str, Any]:
    """
    Use an LLM to compare baseline (no RAG) vs RAG answers and determine which performs better.
    
    Returns a dict with:
    - winner: str ("baseline", "rag", or "tie")
    - baseline_score: float (0.0 to 1.0)
    - rag_score: float (0.0 to 1.0)
    - reasoning: str
    - rag_improved: bool
    """
    if not model:
        model = os.getenv("JUDGE_LLM_MODEL") or "Mistral-7B-Instruct"
    
    prompt = f"""You are an expert evaluator. Compare two answers to determine which one better answers the question.

Question: {question}

Reference Answer (Ground Truth): {reference_answer}

Answer 1 (Baseline - No RAG): {baseline_answer}

Answer 2 (With RAG): {rag_answer}

Evaluate both answers on:
1. Correctness: Does it answer the question correctly?
2. Completeness: Does it cover the key points from the reference?
3. Accuracy: Are the facts correct?
4. Relevance: How well does it address the question?

Compare the two answers and determine:
- Which answer is better overall?
- Did RAG improve the answer quality?

Respond in JSON format:
{{
    "winner": <"baseline", "rag", or "tie">,
    "baseline_score": <float between 0.0 and 1.0>,
    "rag_score": <float between 0.0 and 1.0>,
    "rag_improved": <true or false>,
    "reasoning": "<brief explanation of comparison>"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=600,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            winner = result.get("winner", "tie").lower()
            if winner not in ["baseline", "rag", "tie"]:
                winner = "tie"
            return {
                "winner": winner,
                "baseline_score": float(result.get("baseline_score", 0.0)),
                "rag_score": float(result.get("rag_score", 0.0)),
                "rag_improved": bool(result.get("rag_improved", False)),
                "reasoning": result.get("reasoning", ""),
            }
        else:
            # Fallback: try to determine winner from text
            content_lower = content.lower()
            if "rag" in content_lower and "better" in content_lower:
                winner = "rag"
                rag_improved = True
            elif "baseline" in content_lower and "better" in content_lower:
                winner = "baseline"
                rag_improved = False
            else:
                winner = "tie"
                rag_improved = False
            
            return {
                "winner": winner,
                "baseline_score": 0.5,
                "rag_score": 0.5,
                "rag_improved": rag_improved,
                "reasoning": content[:200],
            }
    except Exception as e:
        print(f"WARNING: LLM comparison failed: {e}", file=sys.stderr)
        return {
            "winner": "tie",
            "baseline_score": 0.0,
            "rag_score": 0.0,
            "rag_improved": False,
            "reasoning": f"Error: {str(e)}",
        }


def run_rag_evaluation_if_needed(input_xlsx: Path, output_xlsx: Path) -> Path:
    """
    Run RAG evaluation if rag_answer column doesn't exist.
    
    Returns the path to the file with RAG answers.
    """
    df = pd.read_excel(input_xlsx)
    
    # Check if RAG evaluation already done
    if "rag_answer" in df.columns and df["rag_answer"].notna().any():
        print(f"✓ RAG answers already present in {input_xlsx}")
        return input_xlsx
    
    print(f"Running RAG evaluation...")
    print(f"  Input: {input_xlsx}")
    print(f"  Output: {output_xlsx}")
    
    # Use the existing warc-rag CLI
    import subprocess
    cmd = [
        sys.executable, "-m", "warc_tools.rag.cli",
        "--eval-xlsx", str(input_xlsx), str(output_xlsx)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: RAG evaluation failed", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ RAG evaluation complete")
    return output_xlsx


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG and baseline answers with multiple metrics"
    )
    parser.add_argument(
        "input_xlsx",
        type=Path,
        help="Input Excel file with questions, answers, baseline_answer, and optionally rag_answer"
    )
    parser.add_argument(
        "output_xlsx",
        type=Path,
        help="Output Excel file with evaluation results"
    )
    parser.add_argument(
        "--run-rag",
        action="store_true",
        help="Run RAG evaluation if rag_answer column is missing"
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster, but less comprehensive)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Model to use for LLM-as-judge (default: Mistral-7B-Instruct)"
    )
    
    args = parser.parse_args()
    
    # Load input file
    if not args.input_xlsx.exists():
        print(f"ERROR: Input file not found: {args.input_xlsx}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_excel(args.input_xlsx)
    
    # Check required columns
    required = ["question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    # Run RAG evaluation if needed
    if args.run_rag or "rag_answer" not in df.columns:
        rag_output = args.output_xlsx.parent / f"{args.output_xlsx.stem}_rag_temp.xlsx"
        rag_file = run_rag_evaluation_if_needed(args.input_xlsx, rag_output)
        df = pd.read_excel(rag_file)
        if rag_file != args.input_xlsx:
            # Clean up temp file
            rag_file.unlink()
    
    # Check for baseline_answer
    if "baseline_answer" not in df.columns:
        print("WARNING: baseline_answer column not found. Skipping baseline evaluation.", file=sys.stderr)
        df["baseline_answer"] = ""
    
    # Check for rag_answer
    if "rag_answer" not in df.columns:
        print("ERROR: rag_answer column not found. Run with --run-rag flag.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nEvaluating {len(df)} questions...")
    
    # Initialize evaluation columns
    df["exact_match_baseline"] = False
    df["exact_match_rag"] = False
    df["multiple_choice_baseline"] = None
    df["multiple_choice_rag"] = None
    df["llm_judge_score_baseline"] = None
    df["llm_judge_correct_baseline"] = None
    df["llm_judge_reasoning_baseline"] = None
    df["llm_judge_score_rag"] = None
    df["llm_judge_correct_rag"] = None
    df["llm_judge_reasoning_rag"] = None
    # Comparison columns
    df["llm_compare_winner"] = None
    df["llm_compare_baseline_score"] = None
    df["llm_compare_rag_score"] = None
    df["llm_compare_rag_improved"] = None
    df["llm_compare_reasoning"] = None
    
    # Source comparison columns (compare retrieved URLs with relevant_doc_1 and relevant_doc_2)
    df["source_match_doc1"] = None
    df["source_match_doc2"] = None
    df["source_match_any"] = None
    df["retrieved_source_count"] = None
    
    # Get LLM-as-judge client
    judge_client = None
    if not args.skip_llm_judge:
        try:
            judge_client = get_judge_llm_client()
            judge_model = args.judge_model or os.getenv("JUDGE_LLM_MODEL") or "Mistral-7B-Instruct"
            if not args.judge_model and not os.getenv("JUDGE_LLM_MODEL"):
                print(f"Using default LLM-as-judge model: {judge_model}")
            else:
                print(f"Using LLM-as-judge model: {judge_model}")
        except Exception as e:
            print(f"WARNING: Could not initialize LLM-as-judge: {e}", file=sys.stderr)
            print("Continuing without LLM-as-judge evaluation...", file=sys.stderr)
            args.skip_llm_judge = True
    
    # Progress tracking file
    progress_file = Path(str(args.output_xlsx) + ".progress")
    total_rows = len(df)
    print(f"Evaluating {total_rows} questions...")
    
    # Evaluate each row
    for idx, row in df.iterrows():
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        reference = str(row["answer"]) if pd.notna(row["answer"]) else ""
        baseline = str(row["baseline_answer"]) if pd.notna(row["baseline_answer"]) else ""
        rag = str(row["rag_answer"]) if pd.notna(row["rag_answer"]) else ""
        
        if not question:
            continue
        
        # Print progress
        if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
            print(f"[{idx + 1}/{total_rows}] Processing question {idx + 1}...")
        
        # Exact match
        if reference:
            df.at[idx, "exact_match_baseline"] = compute_exact_match(baseline, reference)
            df.at[idx, "exact_match_rag"] = compute_exact_match(rag, reference)
        
        # Multiple choice
        mc_baseline = compute_multiple_choice_score(baseline, reference)
        mc_rag = compute_multiple_choice_score(rag, reference)
        if mc_baseline is not None:
            df.at[idx, "multiple_choice_baseline"] = mc_baseline
        if mc_rag is not None:
            df.at[idx, "multiple_choice_rag"] = mc_rag
        
        # LLM-as-judge (individual evaluation)
        if judge_client and reference:
            if baseline:
                judge_result = llm_as_judge(
                    judge_client, question, reference, baseline, judge_model
                )
                df.at[idx, "llm_judge_score_baseline"] = judge_result["score"]
                df.at[idx, "llm_judge_correct_baseline"] = judge_result["correct"]
                df.at[idx, "llm_judge_reasoning_baseline"] = judge_result["reasoning"]
            
            if rag:
                judge_result = llm_as_judge(
                    judge_client, question, reference, rag, judge_model
                )
                df.at[idx, "llm_judge_score_rag"] = judge_result["score"]
                df.at[idx, "llm_judge_correct_rag"] = judge_result["correct"]
                df.at[idx, "llm_judge_reasoning_rag"] = judge_result["reasoning"]
        
        # LLM-as-judge comparison (baseline vs RAG)
        if judge_client and reference and baseline and rag:
            compare_result = llm_compare_answers(
                judge_client, question, reference, baseline, rag, judge_model
            )
            df.at[idx, "llm_compare_winner"] = compare_result["winner"]
            df.at[idx, "llm_compare_baseline_score"] = compare_result["baseline_score"]
            df.at[idx, "llm_compare_rag_score"] = compare_result["rag_score"]
            df.at[idx, "llm_compare_rag_improved"] = compare_result["rag_improved"]
            df.at[idx, "llm_compare_reasoning"] = compare_result["reasoning"]
        
        # Source comparison (compare retrieved URLs with relevant_doc_1 and relevant_doc_2)
        if "rag_source_urls" in df.columns and pd.notna(row.get("rag_source_urls")):
            source_urls = str(row["rag_source_urls"]).split("; ") if pd.notna(row.get("rag_source_urls")) else []
            relevant_doc1 = str(row.get("relevant_doc_1", "")) if pd.notna(row.get("relevant_doc_1")) else ""
            relevant_doc2 = str(row.get("relevant_doc_2", "")) if pd.notna(row.get("relevant_doc_2")) else ""
            
            match_doc1 = any(relevant_doc1 in url or url in relevant_doc1 for url in source_urls if url and relevant_doc1)
            match_doc2 = any(relevant_doc2 in url or url in relevant_doc2 for url in source_urls if url and relevant_doc2)
            match_any = match_doc1 or match_doc2
            
            df.at[idx, "source_match_doc1"] = match_doc1
            df.at[idx, "source_match_doc2"] = match_doc2
            df.at[idx, "source_match_any"] = match_any
            df.at[idx, "retrieved_source_count"] = len([u for u in source_urls if u])
        
        # Save progress every 10 questions
        if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
            print(f"\n[{idx + 1}/{total_rows}] Processed {idx + 1} questions...", flush=True)
            try:
                # Save current progress (don't rename columns yet, keep original)
                df.to_excel(progress_file, index=False)
                print(f"  ✓ Progress saved to {progress_file.name}", flush=True)
            except Exception as e:
                print(f"  ⚠️  Could not save progress: {e}", flush=True)
    
    # Rename columns for clarity (as requested by colleague)
    # Keep original columns, but add aliases
    column_mapping = {
        "answer": "golden_answer",
        "baseline_answer": "qwen_answer_wo_rag",
        "rag_answer": "qwen_answer_with_rag",
    }
    
    # Create a copy with renamed columns for the main output
    df_output = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_output.columns:
            df_output[new_name] = df_output[old_name]
    
    # Reorder columns: question, golden_answer, qwen_answer_wo_rag, qwen_answer_with_rag, then metrics
    priority_cols = ["question", "golden_answer", "qwen_answer_wo_rag", "qwen_answer_with_rag"]
    other_cols = [c for c in df_output.columns if c not in priority_cols]
    df_output = df_output[priority_cols + other_cols]
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY STATISTICS")
    print("="*60)
    
    total_questions = len(df_output)
    print(f"\nTotal Questions Evaluated: {total_questions}")
    
    # Exact Match
    if "exact_match_baseline" in df_output.columns:
        em_baseline = df_output["exact_match_baseline"].sum() / total_questions
        em_rag = df_output["exact_match_rag"].sum() / total_questions
        em_improvement = em_rag - em_baseline
        print(f"\n📊 Exact Match Accuracy:")
        print(f"   Baseline (no RAG): {em_baseline:.2%} ({df_output['exact_match_baseline'].sum()}/{total_questions})")
        print(f"   RAG:               {em_rag:.2%} ({df_output['exact_match_rag'].sum()}/{total_questions})")
        if em_improvement > 0:
            print(f"   ✅ RAG Improvement:  +{em_improvement:.2%} ({em_improvement*100:.1f} percentage points)")
        elif em_improvement < 0:
            print(f"   ⚠️  RAG Change:       {em_improvement:.2%} ({em_improvement*100:.1f} percentage points)")
        else:
            print(f"   ➡️  No change")
    
    # Multiple Choice
    if "multiple_choice_baseline" in df_output.columns:
        mc_valid = df_output["multiple_choice_baseline"].notna().sum()
        if mc_valid > 0:
            mc_baseline = df_output["multiple_choice_baseline"].sum() / mc_valid
            mc_rag = df_output["multiple_choice_rag"].sum() / df_output["multiple_choice_rag"].notna().sum() if df_output["multiple_choice_rag"].notna().any() else 0
            mc_improvement = mc_rag - mc_baseline
            print(f"\n📊 Multiple Choice Accuracy ({mc_valid} applicable questions):")
            print(f"   Baseline (no RAG): {mc_baseline:.2%} ({df_output['multiple_choice_baseline'].sum()}/{mc_valid})")
            print(f"   RAG:               {mc_rag:.2%} ({df_output['multiple_choice_rag'].sum()}/{mc_valid})")
            if mc_improvement > 0:
                print(f"   ✅ RAG Improvement:  +{mc_improvement:.2%}")
    
    # LLM Judge Scores
    if "llm_judge_score_baseline" in df_output.columns:
        judge_baseline = df_output["llm_judge_score_baseline"].mean() if df_output["llm_judge_score_baseline"].notna().any() else None
        judge_rag = df_output["llm_judge_score_rag"].mean() if df_output["llm_judge_score_rag"].notna().any() else None
        if judge_baseline is not None and judge_rag is not None:
            judge_improvement = judge_rag - judge_baseline
            print(f"\n📊 LLM-as-Judge Quality Score (0.0-1.0):")
            print(f"   Baseline (no RAG): {judge_baseline:.3f}")
            print(f"   RAG:               {judge_rag:.3f}")
            if judge_improvement > 0:
                print(f"   ✅ RAG Improvement:  +{judge_improvement:.3f} ({judge_improvement*100:.1f} points)")
            elif judge_improvement < 0:
                print(f"   ⚠️  RAG Change:       {judge_improvement:.3f}")
            else:
                print(f"   ➡️  No change")
    
    # RAG vs Baseline Comparison
    if "llm_compare_rag_improved" in df_output.columns:
        rag_improved_count = df_output["llm_compare_rag_improved"].sum() if df_output["llm_compare_rag_improved"].notna().any() else 0
        rag_improved_total = df_output["llm_compare_rag_improved"].notna().sum()
        if rag_improved_total > 0:
            rag_improved_pct = rag_improved_count / rag_improved_total
            print(f"\n📊 RAG vs Baseline Direct Comparison ({rag_improved_total} compared):")
            print(f"   ✅ RAG Improved:     {rag_improved_pct:.2%} ({rag_improved_count}/{rag_improved_total} questions)")
            if "llm_compare_winner" in df_output.columns:
                winner_counts = df_output["llm_compare_winner"].value_counts()
                print(f"   Winner Distribution:")
                for winner, count in winner_counts.items():
                    pct = count / rag_improved_total
                    icon = "✅" if winner == "rag" else "➡️" if winner == "tie" else "⚠️"
                    print(f"     {icon} {winner}: {pct:.2%} ({count} questions)")
    
    # Source Retrieval Quality
    if "source_match_any" in df_output.columns:
        source_match_count = df_output["source_match_any"].sum() if df_output["source_match_any"].notna().any() else 0
        source_match_total = df_output["source_match_any"].notna().sum()
        if source_match_total > 0:
            source_match_pct = source_match_count / source_match_total
            avg_retrieved = df_output["retrieved_source_count"].mean() if "retrieved_source_count" in df_output.columns else 0
            print(f"\n📊 Source Retrieval Quality:")
            print(f"   ✅ Retrieved Relevant Doc: {source_match_pct:.2%} ({source_match_count}/{source_match_total} questions)")
            print(f"   Average Sources Retrieved: {avg_retrieved:.1f} per question")
            if "source_match_doc1" in df_output.columns:
                doc1_match = df_output["source_match_doc1"].sum() if df_output["source_match_doc1"].notna().any() else 0
                doc2_match = df_output["source_match_doc2"].sum() if df_output["source_match_doc2"].notna().any() else 0
                print(f"   Matched relevant_doc_1:   {doc1_match}/{source_match_total} ({doc1_match/source_match_total:.2%})")
                print(f"   Matched relevant_doc_2:   {doc2_match}/{source_match_total} ({doc2_match/source_match_total:.2%})")
    
    print("\n" + "="*60)
    
    # Save output
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_excel(args.output_xlsx, index=False)
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()
        print(f"  Cleaned up progress file")
    
    print(f"\n✓ Evaluation results saved to: {args.output_xlsx}")


if __name__ == "__main__":
    main()

