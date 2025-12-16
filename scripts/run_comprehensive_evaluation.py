#!/usr/bin/env python3
"""
Comprehensive evaluation script for Qwen and Apertus models.

This script:
1. Runs baseline (no RAG) for Qwen and Apertus
2. Runs RAG evaluation for Qwen and Apertus
3. Computes evaluation metrics (Exact Match, Multiple Choice, LLM-as-Judge)
4. Creates two separate Excel files with filtered results
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Model configurations
# Note: User launched Qwen/Qwen3-Next-80B-A3B-Instruct on cluster
# If you need a different model name, use --qwen-model flag
QWEN_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
APERTUS_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"

def run_baseline(input_xlsx: Path, output_xlsx: Path, model_name: str, force_regenerate: bool = False) -> None:
    """Run baseline evaluation (no RAG) for a model."""
    print(f"\n{'='*60}")
    print(f"Running BASELINE evaluation for {model_name}")
    print(f"{'='*60}")
    
    # Check if baseline already exists and matches the model
    if not force_regenerate and input_xlsx.exists():
        try:
            df = pd.read_excel(input_xlsx)
            if "baseline_answer" in df.columns and "baseline_model" in df.columns:
                existing_model = df["baseline_model"].iloc[0] if len(df) > 0 else None
                if existing_model == model_name and df["baseline_answer"].notna().any():
                    print(f"[OK] Baseline answers already exist for {model_name}")
                    print(f"  Copying existing baseline to output...")
                    df.to_excel(output_xlsx, index=False)
                    return
        except Exception as e:
            print(f"  Warning: Could not check existing baseline: {e}")
            print(f"  Will regenerate...")
    
    cmd = [
        sys.executable, "-m", "warc_tools.baseline.cli",
        str(input_xlsx),
        str(output_xlsx),
        model_name
    ]
    
    env = os.environ.copy()
    # Baseline CLI uses CSCS_API_KEY, map from LLM_API_KEY if needed
    llm_key = env.get("LLM_API_KEY") or env.get("EMBED_API_KEY")
    if not env.get("CSCS_API_KEY") and llm_key:
        env["CSCS_API_KEY"] = llm_key
        print(f"  Using LLM_API_KEY as CSCS_API_KEY")
    if not env.get("CSCS_BASE_URL"):
        env["CSCS_BASE_URL"] = env.get("LLM_BASE_URL") or env.get("EMBED_BASE_URL") or "https://api.swissai.cscs.ch/v1"
        print(f"  Using base URL: {env['CSCS_BASE_URL']}")
    
    if not env.get("CSCS_API_KEY"):
        print(f"ERROR: CSCS_API_KEY not set. Please set LLM_API_KEY or EMBED_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Baseline evaluation failed for {model_name}")
        print(result.stderr)
        sys.exit(1)
    
    print(f"[OK] Baseline evaluation complete: {output_xlsx}")

def run_rag_evaluation(input_xlsx: Path, output_xlsx: Path, model_name: str) -> None:
    """Run RAG evaluation for a model."""
    print(f"\n{'='*60}")
    print(f"Running RAG evaluation for {model_name}")
    print(f"{'='*60}")
    
    # Create environment with updated LLM_MODEL
    env = os.environ.copy()
    env["LLM_MODEL"] = model_name
    
    # Also set CSCS_API_KEY if using CSCS provider
    if not env.get("LLM_API_KEY") and env.get("EMBED_API_KEY"):
        env["LLM_API_KEY"] = env["EMBED_API_KEY"]
    
    cmd = [
        sys.executable, "-m", "warc_tools.rag.cli",
        "--eval-xlsx",
        str(input_xlsx),
        str(output_xlsx)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Using LLM_MODEL={model_name}")
    
    # Run with real-time output for monitoring
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Print output for debugging
    if result.stdout:
        print("STDOUT (last 500 chars):")
        print(result.stdout[-500:])
    
    if result.returncode != 0:
        print(f"ERROR: RAG evaluation failed for {model_name}")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        sys.exit(1)
    
    print(f"[OK] RAG evaluation complete: {output_xlsx}")

def run_metrics_evaluation(input_xlsx: Path, output_xlsx: Path, skip_llm_judge: bool = False) -> None:
    """Run metrics evaluation (Exact Match, Multiple Choice, LLM-as-Judge)."""
    print(f"\n{'='*60}")
    print(f"Computing evaluation metrics")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "scripts/evaluate_answers.py",
        str(input_xlsx),
        str(output_xlsx)
    ]
    
    if skip_llm_judge:
        cmd.append("--skip-llm-judge")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Metrics evaluation failed")
        print(result.stderr)
        sys.exit(1)
    
    print(f"[OK] Metrics evaluation complete: {output_xlsx}")

def filter_bad_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out metrics that give very bad results or are not useful.
    Keeps only meaningful metrics with positive results.
    """
    # Core columns to always keep
    core_cols = [
        "question", "golden_answer", 
        "qwen_answer_wo_rag", "qwen_answer_with_rag",
        "apertus_answer_wo_rag", "apertus_answer_with_rag",
        "rag_source_urls",  # Keep source URLs for comparison
    ]
    
    # Metrics to keep (only useful/positive ones)
    useful_metrics = [
        # Exact Match (always useful)
        "exact_match_baseline", "exact_match_rag",
        
        # Multiple Choice (useful if applicable)
        "multiple_choice_baseline", "multiple_choice_rag",
        
        # LLM Judge scores (useful - keep scores, not just correct/incorrect)
        "llm_judge_score_baseline", "llm_judge_score_rag",
        "llm_judge_correct_baseline", "llm_judge_correct_rag",
        
        # LLM Compare (useful for RAG vs baseline)
        "llm_compare_winner", "llm_compare_baseline_score", "llm_compare_rag_score",
        "llm_compare_rag_improved",
        
        # Source comparison (NEW - very useful!)
        "source_match_doc1", "source_match_doc2", "source_match_any",
        "retrieved_source_count",
        
        # Original metadata (keep for reference)
        "lang", "relevant_doc_1", "relevant_doc_2", "baseline_model",
    ]
    
    # Filter: keep core + useful metrics + any columns that don't look like bad metrics
    keep_cols = []
    
    # Add core columns
    for col in core_cols:
        if col in df.columns:
            keep_cols.append(col)
    
    # Add useful metrics
    for col in useful_metrics:
        if col in df.columns:
            keep_cols.append(col)
    
    # Add other columns that are not reasoning/verbose (but keep them if they're short)
    for col in df.columns:
        if col not in keep_cols:
            # Skip verbose reasoning columns
            if "reasoning" in col.lower() and len(str(df[col].iloc[0] if len(df) > 0 else "")) > 200:
                continue
            # Keep other metadata columns
            if col not in ["llm_judge_reasoning_baseline", "llm_judge_reasoning_rag", 
                          "llm_compare_reasoning"]:
                keep_cols.append(col)
    
    return df[[c for c in keep_cols if c in df.columns]]

def create_model_specific_output(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Create model-specific output with renamed columns."""
    df_output = df.copy()
    
    if model_name.lower().startswith("qwen"):
        # Qwen output: question, golden_answer, qwen_answer_wo_rag, qwen_answer_with_rag
        if "baseline_answer" in df_output.columns:
            df_output["qwen_answer_wo_rag"] = df_output["baseline_answer"]
        if "rag_answer" in df_output.columns:
            df_output["qwen_answer_with_rag"] = df_output["rag_answer"]
        
        # Rename metrics columns to be Qwen-specific
        metric_rename = {
            "exact_match_baseline": "qwen_exact_match_wo_rag",
            "exact_match_rag": "qwen_exact_match_with_rag",
            "multiple_choice_baseline": "qwen_multiple_choice_wo_rag",
            "multiple_choice_rag": "qwen_multiple_choice_with_rag",
            "llm_judge_score_baseline": "qwen_llm_judge_score_wo_rag",
            "llm_judge_score_rag": "qwen_llm_judge_score_with_rag",
            "llm_judge_correct_baseline": "qwen_llm_judge_correct_wo_rag",
            "llm_judge_correct_rag": "qwen_llm_judge_correct_with_rag",
        }
        
    else:  # Apertus
        # Apertus output: question, golden_answer, apertus_answer_wo_rag, apertus_answer_with_rag
        if "baseline_answer" in df_output.columns:
            df_output["apertus_answer_wo_rag"] = df_output["baseline_answer"]
        if "rag_answer" in df_output.columns:
            df_output["apertus_answer_with_rag"] = df_output["rag_answer"]
        
        # Rename metrics columns to be Apertus-specific
        metric_rename = {
            "exact_match_baseline": "apertus_exact_match_wo_rag",
            "exact_match_rag": "apertus_exact_match_with_rag",
            "multiple_choice_baseline": "apertus_multiple_choice_wo_rag",
            "multiple_choice_rag": "apertus_multiple_choice_with_rag",
            "llm_judge_score_baseline": "apertus_llm_judge_score_wo_rag",
            "llm_judge_score_rag": "apertus_llm_judge_score_with_rag",
            "llm_judge_correct_baseline": "apertus_llm_judge_correct_wo_rag",
            "llm_judge_correct_rag": "apertus_llm_judge_correct_with_rag",
        }
    
    # Apply renames
    for old_name, new_name in metric_rename.items():
        if old_name in df_output.columns:
            df_output[new_name] = df_output[old_name]
    
    # Reorder priority columns
    if model_name.lower().startswith("qwen"):
        priority_cols = ["question", "golden_answer", "qwen_answer_wo_rag", "qwen_answer_with_rag"]
    else:
        priority_cols = ["question", "golden_answer", "apertus_answer_wo_rag", "apertus_answer_with_rag"]
    
    other_cols = [c for c in df_output.columns if c not in priority_cols]
    df_output = df_output[priority_cols + other_cols]
    
    return df_output

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for Qwen and Apertus models"
    )
    parser.add_argument(
        "input_xlsx",
        type=Path,
        help="Input Excel file with questions and golden answers"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for results (default: evaluation/results)"
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster)"
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=QWEN_MODEL,
        help=f"Qwen model name (default: {QWEN_MODEL})"
    )
    parser.add_argument(
        "--apertus-model",
        type=str,
        default=APERTUS_MODEL,
        help=f"Apertus model name (default: {APERTUS_MODEL})"
    )
    
    args = parser.parse_args()
    
    if not args.input_xlsx.exists():
        print(f"ERROR: Input file not found: {args.input_xlsx}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {args.input_xlsx}")
    print(f"Output directory: {args.output_dir}")
    print(f"Qwen model: {args.qwen_model}")
    print(f"Apertus model: {args.apertus_model}")
    
    # ========================================================================
    # STEP 1: QWEN EVALUATION
    # ========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: QWEN EVALUATION")
    print(f"{'='*60}")
    
    qwen_baseline_file = args.output_dir / "qwen_baseline.xlsx"
    qwen_rag_file = args.output_dir / "qwen_rag.xlsx"
    qwen_metrics_file = args.output_dir / "qwen_metrics.xlsx"
    qwen_final_file = args.output_dir / "qwen_evaluation_final.xlsx"
    
    # 1.1: Qwen baseline (no RAG)
    # Force regenerate since existing baseline is from Qwen/Qwen3-8B, not the new model
    print(f"\nNote: Existing baseline in Excel is from Qwen/Qwen3-8B")
    print(f"      Regenerating with {args.qwen_model}...")
    run_baseline(args.input_xlsx, qwen_baseline_file, args.qwen_model, force_regenerate=True)
    
    # 1.2: Qwen RAG
    run_rag_evaluation(qwen_baseline_file, qwen_rag_file, args.qwen_model)
    
    # 1.3: Qwen metrics
    run_metrics_evaluation(qwen_rag_file, qwen_metrics_file, args.skip_llm_judge)
    
    # 1.4: Filter and format Qwen results
    df_qwen = pd.read_excel(qwen_metrics_file)
    df_qwen = filter_bad_metrics(df_qwen)
    df_qwen = create_model_specific_output(df_qwen, "qwen")
    df_qwen.to_excel(qwen_final_file, index=False)
    print(f"\n[OK] Qwen final results: {qwen_final_file}")
    
    # ========================================================================
    # STEP 2: APERTUS EVALUATION
    # ========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: APERTUS EVALUATION")
    print(f"{'='*60}")
    
    apertus_baseline_file = args.output_dir / "apertus_baseline.xlsx"
    apertus_rag_file = args.output_dir / "apertus_rag.xlsx"
    apertus_metrics_file = args.output_dir / "apertus_metrics.xlsx"
    apertus_final_file = args.output_dir / "apertus_evaluation_final.xlsx"
    
    # 2.1: Apertus baseline (no RAG)
    run_baseline(args.input_xlsx, apertus_baseline_file, args.apertus_model)
    
    # 2.2: Apertus RAG
    run_rag_evaluation(apertus_baseline_file, apertus_rag_file, args.apertus_model)
    
    # 2.3: Apertus metrics
    run_metrics_evaluation(apertus_rag_file, apertus_metrics_file, args.skip_llm_judge)
    
    # 2.4: Filter and format Apertus results
    df_apertus = pd.read_excel(apertus_metrics_file)
    df_apertus = filter_bad_metrics(df_apertus)
    df_apertus = create_model_specific_output(df_apertus, "apertus")
    df_apertus.to_excel(apertus_final_file, index=False)
    print(f"\n[OK] Apertus final results: {apertus_final_file}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nFinal output files:")
    print(f"  1. Qwen:   {qwen_final_file}")
    print(f"  2. Apertus: {apertus_final_file}")
    print(f"\nIntermediate files in: {args.output_dir}")

if __name__ == "__main__":
    main()

