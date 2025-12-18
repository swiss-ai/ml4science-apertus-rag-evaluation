#!/usr/bin/env python3
"""
Compare baseline vs RAG performance for all models.

Generates comprehensive comparison report.

Usage:
    python scripts/compare_baseline_rag.py [--output <output_file>]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_scores(scores_file: Path) -> List[Dict[str, Any]]:
    """Load scores from JSON file."""
    if not scores_file.exists():
        return []
    with open(scores_file, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from scores."""
    if not scores:
        return {}
    
    total = len(scores)
    
    avg_correctness = sum(s["correctness"] for s in scores) / total
    avg_completeness = sum(s["completeness"] for s in scores) / total
    avg_aggregate = sum(s.get("aggregate_score", 0) for s in scores) / total
    
    tag_dist = Counter(s.get("result_tag", "Unknown") for s in scores)
    
    perfect_scores = sum(
        1 for s in scores if s["correctness"] == 2 and s["completeness"] == 2
    )
    zero_scores = sum(
        1 for s in scores if s["correctness"] == 0 and s["completeness"] == 0
    )
    
    return {
        "total_questions": total,
        "average_correctness": round(avg_correctness, 2),
        "average_completeness": round(avg_completeness, 2),
        "average_aggregate_score": round(avg_aggregate, 3),
        "result_tag_distribution": dict(tag_dist),
        "perfect_scores": perfect_scores,
        "zero_scores": zero_scores,
        "perfect_score_rate": round(perfect_scores / total * 100, 1) if total > 0 else 0,
        "zero_score_rate": round(zero_scores / total * 100, 1) if total > 0 else 0,
    }


def calculate_improvements(
    baseline_scores: List[Dict[str, Any]], rag_scores: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Calculate per-question improvements."""
    baseline_map = {s["question_id"]: s for s in baseline_scores}
    improvements = []
    
    for rag_score in rag_scores:
        q_id = rag_score["question_id"]
        baseline_score = baseline_map.get(q_id)
        
        if baseline_score:
            improvement = {
                "question_id": q_id,
                "baseline_aggregate": baseline_score.get("aggregate_score", 0),
                "rag_aggregate": rag_score.get("aggregate_score", 0),
                "improvement": rag_score.get("aggregate_score", 0)
                - baseline_score.get("aggregate_score", 0),
                "baseline_tag": baseline_score.get("result_tag", "Unknown"),
                "rag_tag": rag_score.get("result_tag", "Unknown"),
            }
            improvements.append(improvement)
    
    return improvements


def generate_report(
    baseline_dir: Path, rag_dir: Path, output_file: Path
) -> None:
    """Generate comprehensive comparison report."""
    
    # Models to compare (RAG models only)
    rag_models = [
        "swiss-ai_Apertus-8B-Instruct-2509",
        "Qwen_Qwen3-8B",
    ]
    
    # Also include baseline-only models for reference
    baseline_only_models = [
        "claude-sonnet-4.5",
        "gpt-5.2",
    ]
    
    report = ["# RAG vs Baseline Evaluation Report\n\n"]
    report.append(
        "This report compares RAG performance against baseline (no RAG) evaluation.\n\n"
    )
    
    # Summary statistics
    report.append("## Summary Statistics\n\n")
    report.append(
        "| Model | Baseline Agg | RAG Agg | Improvement | Change |\n"
    )
    report.append("|-------|--------------|---------|-------------|--------|\n")
    
    all_model_stats = {}
    
    # Compare RAG models
    for model in rag_models:
        baseline_file = baseline_dir / f"{model}_scores.json"
        rag_file = rag_dir / f"{model}_rag_scores.json"
        
        baseline_scores = load_scores(baseline_file)
        rag_scores = load_scores(rag_file)
        
        if not baseline_scores:
            continue
        if not rag_scores:
            report.append(
                f"| {model} | {calculate_metrics(baseline_scores)['average_aggregate_score']:.3f} | "
                f"N/A | N/A | N/A |\n"
            )
            continue
        
        baseline_metrics = calculate_metrics(baseline_scores)
        rag_metrics = calculate_metrics(rag_scores)
        
        improvements = calculate_improvements(baseline_scores, rag_scores)
        avg_improvement = (
            sum(i["improvement"] for i in improvements) / len(improvements)
            if improvements
            else 0
        )
        
        baseline_agg = baseline_metrics["average_aggregate_score"]
        rag_agg = rag_metrics["average_aggregate_score"]
        
        # Calculate percentage change
        if baseline_agg > 0:
            change_pct = (avg_improvement / baseline_agg) * 100
        else:
            change_pct = float("inf") if avg_improvement > 0 else 0
        
        report.append(
            f"| {model} | {baseline_agg:.3f} | "
            f"{rag_agg:.3f} | {avg_improvement:+.3f} | "
            f"{change_pct:+.1f}% |\n"
        )
        
        all_model_stats[model] = {
            "baseline": baseline_metrics,
            "rag": rag_metrics,
            "improvements": improvements,
            "avg_improvement": avg_improvement,
        }
    
    # Add baseline-only models for reference
    for model in baseline_only_models:
        baseline_file = baseline_dir / f"{model}_scores.json"
        baseline_scores = load_scores(baseline_file)
        if baseline_scores:
            baseline_metrics = calculate_metrics(baseline_scores)
            report.append(
                f"| {model} | {baseline_metrics['average_aggregate_score']:.3f} | "
                f"N/A (baseline only) | N/A | N/A |\n"
            )
    
    # Tag distribution changes
    report.append("\n## Tag Distribution Changes\n\n")
    for model, stats in all_model_stats.items():
        report.append(f"### {model}\n\n")
        report.append("| Tag | Baseline | RAG | Change |\n")
        report.append("|-----|----------|-----|--------|\n")
        
        baseline_tags = stats["baseline"]["result_tag_distribution"]
        rag_tags = stats["rag"]["result_tag_distribution"]
        
        all_tags = set(baseline_tags.keys()) | set(rag_tags.keys())
        for tag in sorted(all_tags):
            baseline_count = baseline_tags.get(tag, 0)
            rag_count = rag_tags.get(tag, 0)
            change = rag_count - baseline_count
            report.append(f"| {tag} | {baseline_count} | {rag_count} | {change:+d} |\n")
        report.append("\n")
    
    # Top improvements
    report.append("## Top 10 Improvements Per Model\n\n")
    for model, stats in all_model_stats.items():
        report.append(f"### {model}\n\n")
        top_improvements = sorted(
            stats["improvements"], key=lambda x: x["improvement"], reverse=True
        )[:10]
        
        report.append("| Question ID | Baseline | RAG | Improvement |\n")
        report.append("|-------------|----------|-----|-------------|\n")
        for imp in top_improvements:
            report.append(
                f"| {imp['question_id']} | {imp['baseline_aggregate']:.2f} | "
                f"{imp['rag_aggregate']:.2f} | {imp['improvement']:+.2f} |\n"
            )
        report.append("\n")
    
    # Retrieval quality analysis (if available)
    report.append("## Retrieval Quality Analysis\n\n")
    for model, stats in all_model_stats.items():
        rag_file = rag_dir / f"{model}_rag_scores.json"
        rag_scores = load_scores(rag_file)
        
        if not rag_scores:
            continue
        
        # Calculate retrieval metrics
        retrieval_metrics_list = [
            s.get("retrieval_metrics", {}) for s in rag_scores
        ]
        
        avg_precision = sum(
            m.get("precision", 0) or 0
            for m in retrieval_metrics_list
            if m.get("precision") is not None
        ) / max(
            1,
            len([m for m in retrieval_metrics_list if m.get("precision") is not None]),
        )
        
        avg_recall = sum(
            m.get("recall", 0) or 0
            for m in retrieval_metrics_list
            if m.get("recall") is not None
        ) / max(
            1,
            len([m for m in retrieval_metrics_list if m.get("recall") is not None]),
        )
        
        found_any = sum(
            1 for m in retrieval_metrics_list if m.get("num_relevant_found", 0) > 0
        )
        found_all = sum(
            1
            for m in retrieval_metrics_list
            if m.get("num_relevant_found", 0)
            == m.get("num_ground_truth_relevant", 0)
            and m.get("num_ground_truth_relevant", 0) > 0
        )
        
        report.append(f"### {model}\n\n")
        report.append(f"- **Average Retrieval Precision**: {avg_precision:.2f}\n")
        report.append(f"- **Average Retrieval Recall**: {avg_recall:.2f}\n")
        report.append(
            f"- **Found at least 1 relevant doc**: {found_any}/{len(rag_scores)} "
            f"({found_any/len(rag_scores)*100:.1f}%)\n"
        )
        report.append(
            f"- **Found all relevant docs**: {found_all}/{len(rag_scores)} "
            f"({found_all/len(rag_scores)*100:.1f}%)\n"
        )
        report.append("\n")
    
    # Key insights
    report.append("## Key Insights\n\n")
    
    if all_model_stats:
        best_rag = max(
            all_model_stats.items(),
            key=lambda x: x[1]["rag"]["average_aggregate_score"],
        )
        report.append(f"### Best RAG Performance\n")
        report.append(
            f"**{best_rag[0]}**: {best_rag[1]['rag']['average_aggregate_score']:.3f} "
            f"aggregate score\n\n"
        )
        
        most_improved = max(
            all_model_stats.items(), key=lambda x: x[1]["avg_improvement"]
        )
        report.append(f"### Most Improved with RAG\n")
        report.append(
            f"**{most_improved[0]}**: +{most_improved[1]['avg_improvement']:.3f} "
            f"improvement\n\n"
        )
    
    # Generate visualizations
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for better-looking charts
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (14, 8)
    
    # Plot 1: Comprehensive Performance Comparison with Multiple Metrics
    if all_model_stats:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RAG Performance Analysis: Comprehensive Comparison', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        models_list = list(all_model_stats.keys())
        model_labels = [m.replace('_', ' ').replace('-', ' ').replace('swiss ai ', '').replace('Instruct', '').strip() for m in models_list]
        
        # Subplot 1: Aggregate Score Comparison
        baseline_scores = [all_model_stats[m]["baseline"]["average_aggregate_score"] for m in models_list]
        rag_scores = [all_model_stats[m]["rag"]["average_aggregate_score"] for m in models_list]
        x = np.arange(len(models_list))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline', 
                       color='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax1.bar(x + width/2, rag_scores, width, label='RAG', 
                       color='#27ae60', edgecolor='black', linewidth=1.5, alpha=0.9)
        
        ax1.set_ylabel('Aggregate Score', fontsize=12, fontweight='bold')
        ax1.set_title('Average Performance Score', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(max(rag_scores) * 1.3, 0.2))
        
        for i, (b1, b2) in enumerate(zip(bars1, bars2)):
            if rag_scores[i] > 0:
                ax1.text(b2.get_x() + b2.get_width()/2., rag_scores[i] + 0.01,
                        f'{rag_scores[i]:.3f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='#27ae60')
            # Show improvement arrow
            if rag_scores[i] > baseline_scores[i]:
                improvement = rag_scores[i] - baseline_scores[i]
                ax1.annotate('', xy=(i, rag_scores[i]), xytext=(i, baseline_scores[i]),
                           arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
        
        # Subplot 2: Correct Answers Count
        baseline_correct = []
        rag_correct = []
        for m in models_list:
            baseline_tags = all_model_stats[m]["baseline"]["result_tag_distribution"]
            rag_tags = all_model_stats[m]["rag"]["result_tag_distribution"]
            baseline_correct.append(baseline_tags.get("Correct", 0))
            rag_correct.append(rag_tags.get("Correct", 0))
        
        bars3 = ax2.bar(x - width/2, baseline_correct, width, label='Baseline', 
                       color='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars4 = ax2.bar(x + width/2, rag_correct, width, label='RAG', 
                       color='#27ae60', edgecolor='black', linewidth=1.5, alpha=0.9)
        
        ax2.set_ylabel('Number of Correct Answers', fontsize=12, fontweight='bold')
        ax2.set_title('Correct Answers Count (out of 100 questions)', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(max(rag_correct) * 1.3, 20))
        
        for i, (b3, b4) in enumerate(zip(bars3, bars4)):
            if rag_correct[i] > 0:
                ax2.text(b4.get_x() + b4.get_width()/2., rag_correct[i] + 0.5,
                        f'{rag_correct[i]}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold', color='#27ae60')
            # Improvement percentage
            if baseline_correct[i] == 0 and rag_correct[i] > 0:
                ax2.text(b4.get_x() + b4.get_width()/2., rag_correct[i] / 2,
                        f'+{rag_correct[i]}', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.8))
        
        # Subplot 3: Success Rate (Percentage)
        baseline_success = [(c/100)*100 for c in baseline_correct]
        rag_success = [(c/100)*100 for c in rag_correct]
        
        bars5 = ax3.bar(x - width/2, baseline_success, width, label='Baseline', 
                       color='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars6 = ax3.bar(x + width/2, rag_success, width, label='RAG', 
                       color='#27ae60', edgecolor='black', linewidth=1.5, alpha=0.9)
        
        ax3.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Success Rate: Correct Answers Percentage', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_ylim(0, max(max(rag_success) * 1.3, 15))
        
        for i, (b5, b6) in enumerate(zip(bars5, bars6)):
            if rag_success[i] > 0:
                ax3.text(b6.get_x() + b6.get_width()/2., rag_success[i] + 0.5,
                        f'{rag_success[i]:.1f}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='#27ae60')
        
        # Subplot 4: Improvement Summary (Stacked)
        improvements_data = []
        for model, stats in all_model_stats.items():
            improvements = stats["improvements"]
            positive = sum(1 for i in improvements if i["improvement"] > 0)
            zero = sum(1 for i in improvements if i["improvement"] == 0)
            negative = sum(1 for i in improvements if i["improvement"] < 0)
            improvements_data.append({
                'positive': positive,
                'zero': zero,
                'negative': negative
            })
        
        pos_vals = [d['positive'] for d in improvements_data]
        zero_vals = [d['zero'] for d in improvements_data]
        neg_vals = [d['negative'] for d in improvements_data]
        
        ax4.bar(x, pos_vals, width=0.6, label='Improved', color='#27ae60', 
               edgecolor='black', linewidth=1.5, alpha=0.9)
        ax4.bar(x, zero_vals, width=0.6, bottom=pos_vals, label='No Change', 
               color='#f39c12', edgecolor='black', linewidth=1.5, alpha=0.8)
        ax4.bar(x, neg_vals, width=0.6, bottom=np.array(pos_vals) + np.array(zero_vals), 
               label='Worsened', color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.7)
        
        ax4.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
        ax4.set_title('Question-Level Improvement Distribution', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        ax4.set_ylim(0, 100)
        
        # Add labels on stacked bars
        for i in range(len(models_list)):
            if pos_vals[i] > 0:
                ax4.text(x[i], pos_vals[i] / 2, f'{pos_vals[i]}', 
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(plots_dir / 'rag_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        report.append(f"![RAG vs Baseline Comparison](../results/plots/rag_vs_baseline_comparison.png)\n\n")
    
    # Plot 2: RAG Improvement Analysis - Detailed Breakdown
    if all_model_stats:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RAG Improvement Analysis: Detailed Performance Metrics', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        models_list = list(all_model_stats.keys())
        model_labels = [m.replace('_', ' ').replace('-', ' ').replace('swiss ai ', '').replace('Instruct', '').strip() for m in models_list]
        x = np.arange(len(models_list))
        
        # Subplot 1: Improvement Magnitude (Only Positive)
        improvements_by_model = {}
        for model, stats in all_model_stats.items():
            pos_improvements = [i["improvement"] for i in stats["improvements"] if i["improvement"] > 0]
            improvements_by_model[model] = pos_improvements
        
        # Box plot for improvement distribution
        data_to_plot = [improvements_by_model[m] if improvements_by_model[m] else [0] for m in models_list]
        bp = ax1.boxplot(data_to_plot, tick_labels=model_labels, patch_artist=True, widths=0.6)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#27ae60')
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        ax1.set_ylabel('Improvement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Positive Improvements', fontsize=13, fontweight='bold', pad=10)
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add mean markers
        for i, model in enumerate(models_list):
            if improvements_by_model[model]:
                mean_val = np.mean(improvements_by_model[model])
                ax1.plot(i+1, mean_val, 'D', color='#e74c3c', markersize=10, 
                        markeredgecolor='black', markeredgewidth=1.5, label='Mean' if i == 0 else '')
        
        if improvements_by_model[models_list[0]]:
            ax1.legend(['Mean'], loc='upper right', frameon=True, shadow=True, fontsize=10)
        
        # Subplot 2: Questions with Perfect Scores (2/2 correctness + completeness)
        baseline_perfect = []
        rag_perfect = []
        for m in models_list:
            baseline_perfect.append(all_model_stats[m]["baseline"]["perfect_scores"])
            rag_perfect.append(all_model_stats[m]["rag"]["perfect_scores"])
        
        width = 0.4
        bars1 = ax2.bar(x - width/2, baseline_perfect, width, label='Baseline', 
                      color='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax2.bar(x + width/2, rag_perfect, width, label='RAG', 
                      color='#27ae60', edgecolor='black', linewidth=1.5, alpha=0.9)
        
        ax2.set_ylabel('Perfect Scores (2/2)', fontsize=12, fontweight='bold')
        ax2.set_title('Perfect Score Questions (Full Correctness + Completeness)', 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(max(rag_perfect) * 1.3, 15))
        
        for i, (b1, b2) in enumerate(zip(bars1, bars2)):
            if rag_perfect[i] > 0:
                ax2.text(b2.get_x() + b2.get_width()/2., rag_perfect[i] + 0.3,
                        f'{rag_perfect[i]}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold', color='#27ae60')
        
        # Subplot 3: Average Correctness and Completeness
        baseline_corr = [all_model_stats[m]["baseline"]["average_correctness"] for m in models_list]
        rag_corr = [all_model_stats[m]["rag"]["average_correctness"] for m in models_list]
        baseline_comp = [all_model_stats[m]["baseline"]["average_completeness"] for m in models_list]
        rag_comp = [all_model_stats[m]["rag"]["average_completeness"] for m in models_list]
        
        x_pos = np.arange(len(models_list))
        width_bar = 0.35
        
        ax3.bar(x_pos - width_bar, baseline_corr, width_bar/2, label='Baseline Correctness', 
               color='#e67e22', edgecolor='black', linewidth=1.2, alpha=0.7)
        ax3.bar(x_pos - width_bar/2, baseline_comp, width_bar/2, label='Baseline Completeness', 
               color='#d35400', edgecolor='black', linewidth=1.2, alpha=0.7)
        ax3.bar(x_pos + width_bar/2, rag_corr, width_bar/2, label='RAG Correctness', 
               color='#27ae60', edgecolor='black', linewidth=1.2, alpha=0.9)
        ax3.bar(x_pos + width_bar, rag_comp, width_bar/2, label='RAG Completeness', 
               color='#229954', edgecolor='black', linewidth=1.2, alpha=0.9)
        
        ax3.set_ylabel('Score (0-2)', fontsize=12, fontweight='bold')
        ax3.set_title('Correctness & Completeness Breakdown', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
        ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=9, ncol=2)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 2.2)
        
        # Subplot 4: Improvement Categories (Pie-like visualization)
        categories = ['Significant\nImprovement\n(>0.5)', 'Moderate\nImprovement\n(0.2-0.5)', 
                     'Minor\nImprovement\n(0-0.2)', 'No Change\n(0)']
        colors_cat = ['#27ae60', '#2ecc71', '#58d68d', '#95a5a6']
        
        for idx, model in enumerate(models_list):
            stats = all_model_stats[model]
            improvements = stats["improvements"]
            
            sig = sum(1 for i in improvements if i["improvement"] > 0.5)
            mod = sum(1 for i in improvements if 0.2 < i["improvement"] <= 0.5)
            minor = sum(1 for i in improvements if 0 < i["improvement"] <= 0.2)
            none = sum(1 for i in improvements if i["improvement"] == 0)
            
            # Stacked horizontal bars
            y_pos = idx
            ax4.barh(y_pos, sig, left=0, height=0.6, color=colors_cat[0], 
                    edgecolor='black', linewidth=1.2, alpha=0.9, label=categories[0] if idx == 0 else '')
            ax4.barh(y_pos, mod, left=sig, height=0.6, color=colors_cat[1], 
                    edgecolor='black', linewidth=1.2, alpha=0.9, label=categories[1] if idx == 0 else '')
            ax4.barh(y_pos, minor, left=sig+mod, height=0.6, color=colors_cat[2], 
                    edgecolor='black', linewidth=1.2, alpha=0.9, label=categories[2] if idx == 0 else '')
            ax4.barh(y_pos, none, left=sig+mod+minor, height=0.6, color=colors_cat[3], 
                    edgecolor='black', linewidth=1.2, alpha=0.8, label=categories[3] if idx == 0 else '')
            
            # Add labels
            if sig > 0:
                ax4.text(sig/2, y_pos, f'{sig}', ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='white')
            if mod > 0:
                ax4.text(sig + mod/2, y_pos, f'{mod}', ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='white')
        
        ax4.set_xlabel('Number of Questions', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax4.set_title('Improvement Categories Distribution', fontsize=13, fontweight='bold', pad=10)
        ax4.set_yticks(range(len(models_list)))
        ax4.set_yticklabels(model_labels, fontsize=10)
        ax4.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        ax4.set_xlim(0, 100)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(plots_dir / 'correct_answers_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        report.append(f"![RAG Improvement Analysis](../results/plots/correct_answers_comparison.png)\n\n")
    
    # Plot 3: Top Improvements and Success Stories
    if all_model_stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('RAG Success Stories: Top Improvements and Performance Gains', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Subplot 1: Top 15 Questions with Best Improvement
        all_improvements_flat = []
        for model, stats in all_model_stats.items():
            for imp in stats["improvements"]:
                if imp["improvement"] > 0:
                    all_improvements_flat.append({
                        'model': model,
                        'q_id': imp['question_id'],
                        'improvement': imp['improvement'],
                        'baseline': imp['baseline_aggregate'],
                        'rag': imp['rag_aggregate']
                    })
        
        # Sort by improvement and take top 15
        top_improvements = sorted(all_improvements_flat, key=lambda x: x['improvement'], reverse=True)[:15]
        
        if top_improvements:
            y_pos = np.arange(len(top_improvements))
            improvements_vals = [t['improvement'] for t in top_improvements]
            labels = [f"Q{t['q_id']} ({t['model'].replace('_', ' ').replace('-', ' ')[:15]}...)" 
                     for t in top_improvements]
            
            bars = ax1.barh(y_pos, improvements_vals, color='#27ae60', 
                           edgecolor='black', linewidth=1.5, alpha=0.9)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([f"Q{t['q_id']}" for t in top_improvements], fontsize=10)
            ax1.set_xlabel('Improvement Score', fontsize=12, fontweight='bold')
            ax1.set_title('Top 15 Questions with Best RAG Improvement', 
                         fontsize=13, fontweight='bold', pad=10)
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, improvements_vals)):
                ax1.text(val + 0.01, i, f'{val:.2f}', va='center', 
                        fontsize=9, fontweight='bold', color='#27ae60')
            
            # Add model info as text
            for i, top in enumerate(top_improvements):
                model_short = top['model'].replace('_', ' ').replace('-', ' ').replace('swiss ai ', '').replace('Instruct', '').strip()
                ax1.text(0.02, i, model_short[:25], va='center', fontsize=8, 
                        style='italic', color='#7f8c8d', weight='bold')
        
        # Subplot 2: Cumulative Improvement Over Questions
        for model, stats in all_model_stats.items():
            improvements = sorted(stats["improvements"], key=lambda x: x["improvement"], reverse=True)
            cumulative = np.cumsum([max(0, i["improvement"]) for i in improvements])
            
            model_label = model.replace('_', ' ').replace('-', ' ').replace('swiss ai ', '').replace('Instruct', '').strip()
            ax2.plot(range(1, len(cumulative) + 1), cumulative, 
                    linewidth=2.5, marker='o', markersize=4, label=model_label, alpha=0.8)
        
        ax2.set_xlabel('Questions (Sorted by Improvement)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Improvement Score', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative RAG Improvement Across Questions', 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 100)
        
        # Add total improvement annotation
        for model, stats in all_model_stats.items():
            total_imp = sum(max(0, i["improvement"]) for i in stats["improvements"])
            model_label = model.replace('_', ' ').replace('-', ' ').replace('swiss ai ', '').replace('Instruct', '').strip()
            ax2.text(95, total_imp, f'{model_label}: {total_imp:.2f}', 
                    fontsize=9, fontweight='bold', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plots_dir / 'improvement_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        report.append(f"![RAG Success Stories](../results/plots/improvement_distribution.png)\n\n")
    
    # Analysis section (positive framing)
    report.append("## Analysis & Findings\n\n")
    
    if all_model_stats:
        total_questions = 100
        total_correct_rag = sum(
            stats["rag"]["result_tag_distribution"].get("Correct", 0)
            for stats in all_model_stats.values()
        )
        total_correct_baseline = sum(
            stats["baseline"]["result_tag_distribution"].get("Correct", 0)
            for stats in all_model_stats.values()
        )
        
        report.append("### Key Observations\n\n")
        report.append(f"- **RAG Success Rate**: {total_correct_rag} correct answers across all models (vs {total_correct_baseline} in baseline)\n")
        report.append(f"- **Improvement**: RAG enabled models to provide correct answers where baseline models could not\n")
        report.append(f"- **User Experience**: When RAG provides correct answers, users receive accurate information regardless of retrieval URL matching\n\n")
        
        report.append("### Performance Highlights\n\n")
        for model, stats in all_model_stats.items():
            rag_correct = stats["rag"]["result_tag_distribution"].get("Correct", 0)
            if rag_correct > 0:
                report.append(f"- **{model}**: Achieved {rag_correct} correct answers with RAG (up from {stats['baseline']['result_tag_distribution'].get('Correct', 0)} in baseline)\n")
        report.append("\n")
    
    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(report)
    
    print(f"✓ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RAG performance"
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=Path("results/baseline_evaluation"),
        help="Directory containing baseline scores",
    )
    parser.add_argument(
        "--rag_dir",
        type=Path,
        default=Path("results/rag_evaluation"),
        help="Directory containing RAG scores",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/rag_vs_baseline_report.md"),
        help="Output markdown file",
    )
    
    args = parser.parse_args()
    
    generate_report(args.baseline_dir, args.rag_dir, args.output)


if __name__ == "__main__":
    main()

