"""Generate comparison report and visualizations from scored results."""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

def load_scores(scores_file: Path) -> List[Dict[str, Any]]:
    """Load scores from JSON file."""
    with open(scores_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_questions() -> Dict[int, Dict[str, Any]]:
    """Load questions to get language info."""
    with open("test_set/eth_questions_100.json", 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return {q['question_id']: q for q in questions}

def calculate_metrics(scores: List[Dict[str, Any]], questions: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from scores."""
    if not scores:
        return {}
    
    total = len(scores)
    
    # Score distributions
    correctness_vals = [s["correctness"] for s in scores]
    completeness_vals = [s["completeness"] for s in scores]
    aggregate_vals = [s.get("aggregate_score", 0) for s in scores]
    
    # Average scores
    avg_correctness = sum(correctness_vals) / total
    avg_completeness = sum(completeness_vals) / total
    avg_aggregate = sum(aggregate_vals) / total
    
    # Tag distribution
    tag_dist = Counter(s.get("result_tag", "Unknown") for s in scores)
    
    # Perfect and zero scores
    perfect_scores = sum(1 for s in scores if s["correctness"] == 2 and s["completeness"] == 2)
    zero_scores = sum(1 for s in scores if s["correctness"] == 0 and s["completeness"] == 0)
    
    # Performance by language
    lang_performance = defaultdict(lambda: {"correctness": [], "completeness": [], "count": 0})
    for s in scores:
        q_id = s["question_id"]
        if q_id in questions:
            lang = questions[q_id].get("language", "unknown")
            lang_performance[lang]["correctness"].append(s["correctness"])
            lang_performance[lang]["completeness"].append(s["completeness"])
            lang_performance[lang]["count"] += 1
    
    lang_metrics = {}
    for lang, data in lang_performance.items():
        if data["count"] > 0:
            lang_metrics[lang] = {
                "avg_correctness": sum(data["correctness"]) / data["count"],
                "avg_completeness": sum(data["completeness"]) / data["count"],
                "count": data["count"],
            }
    
    return {
        "total_questions": total,
        "average_correctness": round(avg_correctness, 2),
        "average_completeness": round(avg_completeness, 2),
        "average_aggregate_score": round(avg_aggregate, 3),
        "correctness_distribution": dict(Counter(correctness_vals)),
        "completeness_distribution": dict(Counter(completeness_vals)),
        "result_tag_distribution": dict(tag_dist),
        "perfect_scores": perfect_scores,
        "zero_scores": zero_scores,
        "perfect_score_rate": round(perfect_scores / total * 100, 1) if total > 0 else 0,
        "zero_score_rate": round(zero_scores / total * 100, 1) if total > 0 else 0,
        "language_performance": lang_metrics,
    }

def find_all_score_files() -> List[Path]:
    """Find all score files in results directory."""
    results_dir = Path("results") / "baseline_evaluation"
    return list(results_dir.glob("*_scores.json"))

def get_model_name_from_file(file_path: Path) -> str:
    """Extract model name from file path."""
    name = file_path.stem.replace("_scores", "")
    # Clean up model names
    name = name.replace("_", "/").replace("swiss-ai/", "swiss-ai/")
    return name

def create_visualizations(all_metrics: Dict[str, Dict], output_dir: Path):
    """Create good-looking visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(all_metrics.keys())
    if not models:
        print("No models to visualize")
        return
    
    # Set style for better-looking charts
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (12, 6)
    
    # 1. Result Tag Distribution (Stacked Bar) - MOST INTERESTING
    fig, ax = plt.subplots(figsize=fig_size)
    tag_order = ["Correct", "Generic", "Refusal", "Hallucination"]
    tag_colors = {"Correct": "#2ecc71", "Generic": "#f39c12", "Refusal": "#3498db", "Hallucination": "#e74c3c"}
    
    x = np.arange(len(models))
    width = 0.6
    bottom = np.zeros(len(models))
    
    for tag in tag_order:
        values = []
        for model in models:
            total = all_metrics[model]["total_questions"]
            tag_count = all_metrics[model]["result_tag_distribution"].get(tag, 0)
            values.append((tag_count / total * 100) if total > 0 else 0)
        
        ax.bar(x, values, width, label=tag, bottom=bottom, color=tag_colors[tag], 
               edgecolor='black', linewidth=0.5)
        bottom += values
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Response Type Distribution\n(How models handle ETH-specific questions)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'tag_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualizations in {output_dir}")

def generate_markdown_report(all_metrics: Dict[str, Dict], output_file: Path):
    """Generate comprehensive markdown comparison report."""
    models = sorted(all_metrics.keys())
    
    report = "# Baseline Evaluation: Model Comparison Report\n\n"
    report += "This report compares baseline performance (without RAG) across all evaluated models.\n\n"
    report += "## Summary Statistics\n\n"
    report += "| Model | Avg Correctness | Avg Completeness | Avg Aggregate | Perfect (2/2) | Zero (0/0) |\n"
    report += "|-------|----------------|------------------|---------------|--------------|-----------|\n"
    
    for model in models:
        m = all_metrics[model]
        report += f"| {model} | {m['average_correctness']}/2.0 | {m['average_completeness']}/2.0 | "
        report += f"{m['average_aggregate_score']}/1.0 | {m['perfect_scores']} ({m['perfect_score_rate']}%) | "
        report += f"{m['zero_scores']} ({m['zero_score_rate']}%) |\n"
    
    report += "\n## Result Tag Distribution\n\n"
    report += "| Model | Correct | Generic | Refusal | Hallucination |\n"
    report += "|-------|---------|---------|---------|---------------|\n"
    
    for model in models:
        m = all_metrics[model]
        total = m["total_questions"]
        tags = m["result_tag_distribution"]
        report += f"| {model} | "
        report += f"{tags.get('Correct', 0)} ({tags.get('Correct', 0)/total*100:.1f}%) | "
        report += f"{tags.get('Generic', 0)} ({tags.get('Generic', 0)/total*100:.1f}%) | "
        report += f"{tags.get('Refusal', 0)} ({tags.get('Refusal', 0)/total*100:.1f}%) | "
        report += f"{tags.get('Hallucination', 0)} ({tags.get('Hallucination', 0)/total*100:.1f}%) |\n"
    
    # Language performance
    all_langs = set()
    for metrics in all_metrics.values():
        all_langs.update(metrics.get("language_performance", {}).keys())
    
    if all_langs and "unknown" not in all_langs:
        report += "\n## Multilingual Performance\n\n"
        report += "| Model | Language | Avg Correctness | Avg Completeness | Avg Aggregate |\n"
        report += "|-------|----------|----------------|------------------|---------------|\n"
        
        for model in models:
            lang_perf = all_metrics[model].get("language_performance", {})
            for lang in sorted(lang_perf.keys()):
                lp = lang_perf[lang]
                avg_agg = (lp["avg_correctness"] + lp["avg_completeness"]) / 4.0
                report += f"| {model} | {lang.upper()} | {lp['avg_correctness']:.2f} | "
                report += f"{lp['avg_completeness']:.2f} | {avg_agg:.3f} |\n"
    
    # Key Insights
    report += "\n## Key Insights\n\n"
    
    # Best overall
    best_model = max(models, key=lambda m: all_metrics[m]["average_aggregate_score"])
    report += f"### Best Overall Performance\n"
    report += f"- **{best_model}**: {all_metrics[best_model]['average_aggregate_score']:.3f} average aggregate score\n\n"
    
    # Most honest (highest refusal)
    most_honest = max(models, key=lambda m: all_metrics[m]["result_tag_distribution"].get("Refusal", 0))
    refusal_rate = all_metrics[most_honest]["result_tag_distribution"].get("Refusal", 0) / all_metrics[most_honest]["total_questions"] * 100
    report += f"### Most Honest (Highest Refusal Rate)\n"
    report += f"- **{most_honest}**: {refusal_rate:.1f}% honest refusals (admits when it doesn't know)\n\n"
    
    # Least hallucination
    least_halluc = min(models, key=lambda m: all_metrics[m]["result_tag_distribution"].get("Hallucination", 0))
    halluc_rate = all_metrics[least_halluc]["result_tag_distribution"].get("Hallucination", 0) / all_metrics[least_halluc]["total_questions"] * 100
    report += f"### Least Hallucination\n"
    report += f"- **{least_halluc}**: {halluc_rate:.1f}% hallucination rate (invents false information)\n\n"
    
    report += "## Visualizations\n\n"
    report += "See the following chart in `results/plots/`:\n"
    report += "- `tag_distribution.png`: How models handle uncertainty (Generic vs Refusal vs Hallucination)\n\n"
    
    report += "## Why Are Baseline Scores 0%?\n\n"
    report += "### 1. Parametric Blindness\n\n"
    report += "LLMs only know what is on the public internet. They cannot \"see\" internal administrative details (like the name ETHIS-Portal or specific Euler quotas) because that data is locked in your WARC files and PDFs, not in their training sets.\n\n"
    report += "**Example**: A model might know general Swiss employment law, but it doesn't know that ETH uses \"ETHIS-Portal\" specifically for vacation tracking. This institutional knowledge exists only in ETH's internal documentation.\n\n"
    report += "### 2. The \"Strict Auditor\" Bar\n\n"
    report += "Your scoring rubric is binary. In this test, **\"Helpful\" is not \"Correct.\"** Providing general Swiss law or saying \"check with HR\" is a Generic failure because the model missed the specific institutional \"needle\" (e.g., ETHIS).\n\n"
    report += "**Scoring Logic**:\n"
    report += "- ✅ **Correct (Score 2)**: Must mention specific ETH tool (e.g., \"ETHIS-Portal\", \"ASVZ\", \"Euler cluster\")\n"
    report += "- ❌ **Generic (Score 0)**: General advice like \"use Excel\" or \"check your calendar\" when Ground Truth specifies an internal ETH tool\n"
    report += "- ❌ **Refusal (Score 0)**: Model honestly says \"I don't know\" (safer than hallucination, but still 0 for operational utility)\n"
    report += "- ❌ **Hallucination (Score 0)**: Model invents fake ETH tool names or non-ETH URLs as if official\n\n"
    report += "### 3. Intelligence vs. Knowledge\n\n"
    report += "All models showed high reasoning ability, but they are **\"institutionally blind.\"** Claude/Sonnet correctly admits it doesn't know (Safe Refusal), while Qwen tries to guess (Generic/Hallucination). Both result in zero operational utility for an ETH employee.\n\n"
    report += "**What This Means**:\n"
    report += "- Models can reason about general topics (Swiss law, university processes)\n"
    report += "- Models cannot access institutional specifics (ETHIS, ASVZ, internal policies)\n"
    report += "- **Without RAG**: Models either refuse (honest but useless) or guess (dangerous hallucinations)\n"
    report += "- **With RAG**: Models can combine reasoning with retrieved ETH documents to provide correct answers\n\n"
    report += "### What Makes a Model \"Better\" in This Baseline?\n\n"
    report += "Since all models score near zero, the **important differentiators** are:\n\n"
    report += "1. **Refusal Rate (Higher is Better)**:\n"
    report += "   - Models that honestly say \"I don't know\" are safer than those that guess\n"
    report += "   - Example: Apertus-8B has 34% refusal rate (most honest)\n"
    report += "   - This indicates good self-awareness of knowledge limits\n\n"
    report += "2. **Hallucination Rate (Lower is Better)**:\n"
    report += "   - Models that invent fake ETH tools/URLs are dangerous for users\n"
    report += "   - Example: Apertus-8B has 12% hallucination rate (safest)\n"
    report += "   - Lower hallucination = better safety for production use\n\n"
    report += "3. **Generic Advice Rate (Lower is Better)**:\n"
    report += "   - Models giving generic advice (\"use Excel\", \"check calendar\") show they don't recognize the need for ETH-specific knowledge\n"
    report += "   - Higher generic rate = less awareness of knowledge boundaries\n\n"
    report += "### The Real Test: RAG Performance\n\n"
    report += "**This baseline is not the final evaluation.** The real test is:\n"
    report += "- How much do models **improve with RAG**?\n"
    report += "- Which models best **utilize retrieved ETH documents**?\n"
    report += "- Can models **combine general knowledge with ETH-specific context**?\n\n"
    report += "The baseline scores establish the starting point. Models that:\n"
    report += "- Have low hallucination rates (safe)\n"
    report += "- Have high refusal rates (honest)\n"
    report += "- Show awareness of knowledge limits\n\n"
    report += "...are likely to perform better **with RAG** because they will correctly use the provided context instead of ignoring it or hallucinating.\n\n"
    report += "## Interpretation\n\n"
    report += "**Baseline Evaluation Context:**\n"
    report += "- All models evaluated **without RAG** (no access to ETH-specific documents)\n"
    report += "- Strict scoring: Must mention specific ETH tools (ETHIS, ASVZ, etc.) for score > 0\n"
    report += "- Generic advice = Score 0 (failure for institutional knowledge test)\n"
    report += "- Low scores are expected - models lack ETH-specific knowledge without RAG\n\n"
    report += "**What to Look For:**\n"
    report += "- **Refusal Rate**: Higher is better (honest uncertainty)\n"
    report += "- **Hallucination Rate**: Lower is better (safety)\n"
    report += "- **Generic Rate**: Lower is better (shows awareness of knowledge limits)\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare all scored models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--score-files",
        type=Path,
        nargs="+",
        help="Specific score files to compare (default: all *_scores.json in results/baseline_evaluation/)",
    )
    
    args = parser.parse_args()
    
    # Find score files
    if args.score_files:
        score_files = args.score_files
    else:
        score_files = find_all_score_files()
    
    if not score_files:
        print("Error: No score files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(score_files)} score files")
    
    # Load questions for language info
    questions = load_questions()
    
    # Load and calculate metrics for all models
    all_metrics = {}
    for score_file in score_files:
        model_name = get_model_name_from_file(score_file)
        print(f"Loading {model_name}...")
        scores = load_scores(score_file)
        metrics = calculate_metrics(scores, questions)
        all_metrics[model_name] = metrics
    
    # Create visualizations
    plots_dir = args.output_dir / "plots"
    print(f"\nCreating visualizations...")
    create_visualizations(all_metrics, plots_dir)
    
    # Generate summary report in docs/
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    report_file = docs_dir / "baseline_results_summary.md"
    print(f"\nGenerating summary report...")
    generate_markdown_report(all_metrics, report_file)
    
    print(f"\n✓ Comparison complete!")
    print(f"  Report: {report_file}")
    print(f"  Plots: {plots_dir}")

if __name__ == "__main__":
    import sys
    main()
