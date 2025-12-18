"""
MAIN PLOT GENERATION SCRIPT
===========================

This is the PRIMARY script for generating all evaluation plots.

Generated Plots:
- performance_comparison.png: Multi-panel comparison with scores, improvement rates, retrieval quality
- language_analysis.png: Performance analysis by language (English vs German)
- rag_improvement_analysis.png: RAG improvement analysis with winner distribution

Usage:
    python scripts/generate_improved_plots.py

Output:
    All plots saved to: results/plots/

This script generates the exact plots used in the evaluation report.
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Use a more modern style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_data():
    """Load all necessary data."""
    with open("test_set/eth_questions_100.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    baseline_scores = {}
    for model in ["swiss-ai_Apertus-8B-Instruct-2509", "Qwen_Qwen3-Next-80B-A3B-Instruct"]:
        with open(f"results/baseline_evaluation/{model}_scores.json", "r", encoding="utf-8") as f:
            baseline_scores[model] = json.load(f)
    
    rag_scores = {}
    for model in ["swiss-ai_Apertus-8B-Instruct-2509", "Qwen_Qwen3-Next-80B-A3B-Instruct"]:
        with open(f"results/rag_evaluation/{model}_rag_scores.json", "r", encoding="utf-8") as f:
            rag_scores[model] = json.load(f)
    
    question_map = {q["question_id"]: q for q in test_set}
    
    return test_set, baseline_scores, rag_scores, question_map

def plot_performance_comparison(baseline_scores, rag_scores, output_dir):
    """Generate improved performance comparison with better visuals."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Model Performance Comparison: Qwen3-80B vs Apertus-8B', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    models = ["swiss-ai_Apertus-8B-Instruct-2509", "Qwen_Qwen3-Next-80B-A3B-Instruct"]
    model_labels = ["Apertus-8B", "Qwen3-80B"]
    model_colors = {'Apertus-8B': '#e74c3c', 'Qwen3-80B': '#3498db'}
    
    # Subplot 1: LLM-as-Judge Quality Scores (larger, more prominent)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    baseline_agg = [np.mean([s.get("aggregate_score", 0) for s in baseline_scores[m]]) for m in models]
    rag_agg = [np.mean([s.get("aggregate_score", 0) for s in rag_scores[m]]) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_agg, width, label='Baseline (No RAG)', 
                   color='#95a5a6', edgecolor='black', linewidth=2, alpha=0.8)
    bars2 = ax1.bar(x + width/2, rag_agg, width, label='With RAG', 
                   color='#27ae60', edgecolor='black', linewidth=2, alpha=0.95)
    
    # Add gradient effect
    for bar in bars2:
        bar.set_hatch('///')
    
    ax1.set_ylabel('LLM Judge Score', fontsize=14, fontweight='bold')
    ax1.set_title('LLM-as-Judge Quality Scores', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1)
    ax1.set_ylim(0, 1.0)
    
    # Add improvement arrows
    for i in range(len(models)):
        improvement = rag_agg[i] - baseline_agg[i]
        if improvement > 0:
            ax1.annotate('', xy=(i + width/2, rag_agg[i]), xytext=(i - width/2, baseline_agg[i]),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3, alpha=0.7))
            ax1.text(i, (baseline_agg[i] + rag_agg[i])/2, f'+{improvement:.3f}', 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27ae60', linewidth=2))
    
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax1.text(b1.get_x() + b1.get_width()/2., baseline_agg[i] + 0.03,
                f'{baseline_agg[i]:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#34495e')
        ax1.text(b2.get_x() + b2.get_width()/2., rag_agg[i] + 0.03,
                f'{rag_agg[i]:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#27ae60')
    
    # Subplot 2: RAG Improvement Rate (donut chart)
    ax2 = fig.add_subplot(gs[0, 2])
    improvement_rates = []
    for model in models:
        baseline_map = {s["question_id"]: s for s in baseline_scores[model]}
        improved = 0
        total = 0
        for rag_score in rag_scores[model]:
            q_id = rag_score["question_id"]
            baseline_score = baseline_map.get(q_id, {})
            improvement = rag_score.get("aggregate_score", 0) - baseline_score.get("aggregate_score", 0)
            if improvement > 0:
                improved += 1
            total += 1
        rate = (improved / total * 100) if total > 0 else 0
        improvement_rates.append(rate)
    
    # Create donut chart
    colors_donut = ['#27ae60', '#3498db']
    wedges, texts, autotexts = ax2.pie(improvement_rates, labels=model_labels, autopct='%1.1f%%',
                                      colors=colors_donut, startangle=90,
                                      textprops={'fontsize': 11, 'fontweight': 'bold'},
                                      pctdistance=0.85, wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    ax2.set_title('RAG Improvement Rate\n(% questions improved)', 
                 fontsize=13, fontweight='bold', pad=10)
    
    # Subplot 3: Score Change (horizontal bars with annotations)
    ax3 = fig.add_subplot(gs[1, 2])
    score_changes = [rag_agg[i] - baseline_agg[i] for i in range(len(models))]
    
    y_pos = np.arange(len(models))
    colors_change = ['#27ae60' if sc > 0 else '#e74c3c' for sc in score_changes]
    
    bars = ax3.barh(y_pos, score_changes, color=colors_change, 
                   edgecolor='black', linewidth=2, alpha=0.9, height=0.6)
    ax3.set_xlabel('Score Change (RAG - Baseline)', fontsize=12, fontweight='bold')
    ax3.set_title('LLM Judge Score Change', fontsize=13, fontweight='bold', pad=10)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(model_labels, fontsize=11, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, change) in enumerate(zip(bars, score_changes)):
        x_pos = change + 0.01 if change >= 0 else change - 0.01
        ax3.text(x_pos, i, f'{change:+.3f}', ha='left' if change >= 0 else 'right', 
                va='center', fontsize=12, fontweight='bold', color=colors_change[i])
    
    # Subplot 4: Source Retrieval Quality (radar/spider chart style)
    ax4 = fig.add_subplot(gs[2, 0:2])
    retrieval_rates = []
    for model in models:
        total = len(rag_scores[model])
        matched = sum(1 for s in rag_scores[model] 
                     if s.get("retrieval_metrics", {}).get("num_relevant_found", 0) > 0)
        rate = (matched / total * 100) if total > 0 else 0
        retrieval_rates.append(rate)
    
    x_ret = np.arange(len(models))
    bars = ax4.bar(x_ret, retrieval_rates, color=['#e74c3c', '#3498db'], 
                  edgecolor='black', linewidth=2, alpha=0.9, width=0.5)
    ax4.set_ylabel('Match Rate (%)', fontsize=13, fontweight='bold')
    ax4.set_title('Source Retrieval Quality\n(% questions with relevant doc retrieved)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax4.set_xticks(x_ret)
    ax4.set_xticklabels(model_labels, fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax4.set_ylim(0, 100)
    
    for i, (bar, rate) in enumerate(zip(bars, retrieval_rates)):
        ax4.text(i, rate + 3, f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold', color=['#e74c3c', '#3498db'][i])
    
    # Subplot 5: Winner Distribution (stacked horizontal)
    ax5 = fig.add_subplot(gs[2, 2])
    rag_won = []
    baseline_won = []
    for model in models:
        baseline_map = {s["question_id"]: s for s in baseline_scores[model]}
        rag_wins = 0
        baseline_wins = 0
        for rag_score in rag_scores[model]:
            q_id = rag_score["question_id"]
            baseline_score = baseline_map.get(q_id, {})
            rag_agg_val = rag_score.get("aggregate_score", 0)
            baseline_agg_val = baseline_score.get("aggregate_score", 0)
            if rag_agg_val > baseline_agg_val:
                rag_wins += 1
            elif baseline_agg_val > rag_agg_val:
                baseline_wins += 1
        rag_won.append(rag_wins)
        baseline_won.append(baseline_wins)
    
    # Stacked bar
    x_win = np.arange(len(models))
    bars1_win = ax5.barh(x_win, rag_won, color='#27ae60', edgecolor='black', linewidth=2, alpha=0.9, height=0.4)
    bars2_win = ax5.barh(x_win, baseline_won, left=rag_won, color='#95a5a6', 
                         edgecolor='black', linewidth=2, alpha=0.8, height=0.4)
    
    ax5.set_xlabel('Number of Questions', fontsize=12, fontweight='bold')
    ax5.set_title('Winner Distribution', fontsize=13, fontweight='bold', pad=10)
    ax5.set_yticks(x_win)
    ax5.set_yticklabels(model_labels, fontsize=11, fontweight='bold')
    ax5.legend(['RAG Won', 'Baseline Won'], loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (b1, b2) in enumerate(zip(bars1_win, bars2_win)):
        if rag_won[i] > 0:
            ax5.text(rag_won[i]/2, i, f'{rag_won[i]}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='white')
        if baseline_won[i] > 0:
            ax5.text(rag_won[i] + baseline_won[i]/2, i, f'{baseline_won[i]}', 
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated performance_comparison.png")

def plot_language_analysis(test_set, baseline_scores, rag_scores, question_map, output_dir):
    """Generate improved language analysis with better visuals."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Language-Specific Analysis: German vs English Performance', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Get language distribution
    lang_counts = defaultdict(int)
    for q in test_set:
        lang_counts[q.get("language", "en")] += 1
    
    # Subplot 1: Question Language Distribution (enhanced pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    languages = list(lang_counts.keys())
    counts = [lang_counts[lang] for lang in languages]
    colors_pie = ['#3498db', '#2ecc71']
    labels_pie = [f"{lang.upper()}\n({count} questions)" for lang, count in zip(languages, counts)]
    
    wedges, texts, autotexts = ax1.pie(counts, labels=labels_pie, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       explode=(0.05, 0.05), shadow=True, 
                                       wedgeprops=dict(edgecolor='white', linewidth=3))
    ax1.set_title('Question Language Distribution', fontsize=15, fontweight='bold', pad=20)
    
    # Subplot 2: RAG Performance by Language (grouped bars with error bars)
    ax2 = fig.add_subplot(gs[0, 1])
    lang_performance = defaultdict(lambda: {'baseline': [], 'rag': []})
    
    for model in rag_scores.keys():
        for score in rag_scores[model]:
            q_id = score["question_id"]
            lang = question_map.get(q_id, {}).get("language", "en")
            lang_performance[lang]['rag'].append(score.get("aggregate_score", 0))
        
        for score in baseline_scores[model]:
            q_id = score["question_id"]
            lang = question_map.get(q_id, {}).get("language", "en")
            lang_performance[lang]['baseline'].append(score.get("aggregate_score", 0))
    
    langs = sorted(lang_performance.keys())
    rag_avg = [np.mean(lang_performance[lang]['rag']) if lang_performance[lang]['rag'] else 0 for lang in langs]
    baseline_avg = [np.mean(lang_performance[lang]['baseline']) if lang_performance[lang]['baseline'] else 0 for lang in langs]
    rag_std = [np.std(lang_performance[lang]['rag']) if lang_performance[lang]['rag'] else 0 for lang in langs]
    baseline_std = [np.std(lang_performance[lang]['baseline']) if lang_performance[lang]['baseline'] else 0 for lang in langs]
    
    x = np.arange(len(langs))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_avg, width, label='Baseline', 
                   color='#95a5a6', edgecolor='black', linewidth=2, alpha=0.8, yerr=baseline_std, capsize=5)
    bars2 = ax2.bar(x + width/2, rag_avg, width, label='RAG', 
                   color='#27ae60', edgecolor='black', linewidth=2, alpha=0.95, yerr=rag_std, capsize=5)
    
    ax2.set_ylabel('Average LLM Judge Score', fontsize=13, fontweight='bold')
    ax2.set_title('RAG Performance by Language', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lang.upper() for lang in langs], fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax2.set_ylim(0, 1.0)
    
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        if rag_avg[i] > 0:
            ax2.text(b2.get_x() + b2.get_width()/2., rag_avg[i] + rag_std[i] + 0.03,
                    f'{rag_avg[i]:.3f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#27ae60')
    
    # Subplot 3: RAG Improvement by Language (horizontal bars)
    ax3 = fig.add_subplot(gs[1, 0])
    lang_improvements = defaultdict(list)
    
    for model in rag_scores.keys():
        baseline_map = {s["question_id"]: s for s in baseline_scores[model]}
        for rag_score in rag_scores[model]:
            q_id = rag_score["question_id"]
            lang = question_map.get(q_id, {}).get("language", "en")
            baseline_score = baseline_map.get(q_id, {})
            improvement = rag_score.get("aggregate_score", 0) - baseline_score.get("aggregate_score", 0)
            if improvement > 0:
                lang_improvements[lang].append(improvement)
    
    improvement_rates = []
    for lang in langs:
        total_questions = lang_counts[lang]
        improved = len(lang_improvements[lang])
        rate = (improved / total_questions * 100) if total_questions > 0 else 0
        improvement_rates.append(rate)
    
    y_pos = np.arange(len(langs))
    bars = ax3.barh(y_pos, improvement_rates, color=['#3498db', '#2ecc71'], 
                   edgecolor='black', linewidth=2, alpha=0.9, height=0.5)
    ax3.set_xlabel('RAG Improvement Rate (%)', fontsize=13, fontweight='bold')
    ax3.set_title('RAG Improvement by Language\n(% of questions with positive improvement)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([lang.upper() for lang in langs], fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    ax3.set_xlim(0, 100)
    
    for i, (bar, rate) in enumerate(zip(bars, improvement_rates)):
        ax3.text(rate + 2, i, f'{rate:.1f}%', ha='left', va='center', 
                fontsize=12, fontweight='bold', color=['#3498db', '#2ecc71'][i])
    
    # Subplot 4: Source Retrieval by Language (grouped bars)
    ax4 = fig.add_subplot(gs[1, 1])
    lang_retrieval = defaultdict(lambda: {'matched': 0, 'total': 0})
    
    for model in rag_scores.keys():
        for score in rag_scores[model]:
            q_id = score["question_id"]
            lang = question_map.get(q_id, {}).get("language", "en")
            lang_retrieval[lang]['total'] += 1
            metrics = score.get("retrieval_metrics", {})
            if metrics.get("num_relevant_found", 0) > 0:
                lang_retrieval[lang]['matched'] += 1
    
    match_rates = []
    for lang in langs:
        matched = lang_retrieval[lang]['matched']
        total = lang_retrieval[lang]['total']
        rate = (matched / total * 100) if total > 0 else 0
        match_rates.append(rate)
    
    bars = ax4.bar(langs, match_rates, color=['#3498db', '#2ecc71'], 
                  edgecolor='black', linewidth=2, alpha=0.9, width=0.5)
    ax4.set_ylabel('Source Match Rate (%)', fontsize=13, fontweight='bold')
    ax4.set_title('Source Retrieval by Language\n(% questions with relevant doc retrieved)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(range(len(langs)))
    ax4.set_xticklabels([lang.upper() for lang in langs], fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax4.set_ylim(0, 100)
    
    for i, (bar, rate) in enumerate(zip(bars, match_rates)):
        ax4.text(i, rate + 3, f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color=['#3498db', '#2ecc71'][i])
    
    plt.savefig(output_dir / 'language_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated language_analysis.png")

def plot_rag_improvement_analysis(baseline_scores, rag_scores, output_dir):
    """Generate improved RAG improvement analysis."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('RAG Improvement Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    models = ["swiss-ai_Apertus-8B-Instruct-2509", "Qwen_Qwen3-Next-80B-A3B-Instruct"]
    model_labels = ["Apertus-8B", "Qwen3-80B"]
    model_colors = ['#e74c3c', '#9b59b6']
    
    # Subplot 1: RAG vs Baseline: Winner Distribution (pie chart style)
    ax1 = fig.add_subplot(gs[0, 0])
    rag_won = []
    baseline_won = []
    ties = []
    
    for model in models:
        baseline_map = {s["question_id"]: s for s in baseline_scores[model]}
        rag_wins = 0
        baseline_wins = 0
        tie_count = 0
        
        for rag_score in rag_scores[model]:
            q_id = rag_score["question_id"]
            baseline_score = baseline_map.get(q_id, {})
            rag_agg = rag_score.get("aggregate_score", 0)
            baseline_agg = baseline_score.get("aggregate_score", 0)
            
            if rag_agg > baseline_agg:
                rag_wins += 1
            elif baseline_agg > rag_agg:
                baseline_wins += 1
            else:
                tie_count += 1
        
        rag_won.append(rag_wins)
        baseline_won.append(baseline_wins)
        ties.append(tie_count)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, rag_won, width, label='RAG Won', 
                   color='#27ae60', edgecolor='black', linewidth=2, alpha=0.9)
    bars2 = ax1.bar(x, baseline_won, width, label='Baseline Won', 
                   color='#e74c3c', edgecolor='black', linewidth=2, alpha=0.9)
    bars3 = ax1.bar(x + width, ties, width, label='Tie', 
                   color='#95a5a6', edgecolor='black', linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Number of Questions', fontsize=13, fontweight='bold')
    ax1.set_title('RAG vs Baseline: Winner Distribution', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
    
    # Subplot 2: Distribution of RAG Score Improvements (violin plot style)
    ax2 = fig.add_subplot(gs[0, 1])
    all_improvements_by_model = []
    
    for model in models:
        model_improvements = []
        baseline_map = {s["question_id"]: s for s in baseline_scores[model]}
        for rag_score in rag_scores[model]:
            q_id = rag_score["question_id"]
            baseline_score = baseline_map.get(q_id, {})
            improvement = rag_score.get("aggregate_score", 0) - baseline_score.get("aggregate_score", 0)
            model_improvements.append(improvement)
        all_improvements_by_model.append(model_improvements)
    
    # Create histogram with overlay
    bins = np.linspace(-0.3, 1.2, 30)
    for i, (improvements, color) in enumerate(zip(all_improvements_by_model, model_colors)):
        ax2.hist(improvements, bins=bins, alpha=0.6, color=color, 
                edgecolor='black', linewidth=1.5, label=model_labels[i])
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
    ax2.set_ylabel('Number of Questions', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Score Improvement (RAG - Baseline)', fontsize=13, fontweight='bold')
    ax2.set_title('Distribution of RAG Score Improvements', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # Subplot 3: Correct Answers Count (more prominent)
    ax3 = fig.add_subplot(gs[1, 0])
    baseline_correct = []
    rag_correct = []
    
    for model in models:
        baseline_c = sum(1 for s in baseline_scores[model] 
                        if s.get("result_tag") in ["Correct", "Partial"])
        rag_c = sum(1 for s in rag_scores[model] 
                   if s.get("result_tag") in ["Correct", "Partial"])
        baseline_correct.append(baseline_c)
        rag_correct.append(rag_c)
    
    x_corr = np.arange(len(models))
    width_corr = 0.35
    
    bars1_corr = ax3.bar(x_corr - width_corr/2, baseline_correct, width_corr, 
                        label='Baseline', color='#95a5a6', edgecolor='black', 
                        linewidth=2, alpha=0.8)
    bars2_corr = ax3.bar(x_corr + width_corr/2, rag_correct, width_corr, 
                        label='RAG', color='#27ae60', edgecolor='black', 
                        linewidth=2, alpha=0.95)
    
    ax3.set_ylabel('Number of Correct/Partial Answers', fontsize=13, fontweight='bold')
    ax3.set_title('Correct Answers Count', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x_corr)
    ax3.set_xticklabels(model_labels, fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax3.set_ylim(0, 100)
    
    for i, (b1, b2) in enumerate(zip(bars1_corr, bars2_corr)):
        if baseline_correct[i] > 0:
            ax3.text(b1.get_x() + b1.get_width()/2., baseline_correct[i] + 2,
                    f'{baseline_correct[i]}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#34495e')
        if rag_correct[i] > 0:
            ax3.text(b2.get_x() + b2.get_width()/2., rag_correct[i] + 2,
                    f'{rag_correct[i]}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#27ae60')
            # Show improvement
            improvement = rag_correct[i] - baseline_correct[i]
            if improvement > 0:
                ax3.annotate(f'+{improvement}', 
                            xy=(i, rag_correct[i]), 
                            xytext=(i, rag_correct[i] + 5),
                            ha='center', fontsize=12, fontweight='bold', color='#27ae60',
                            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    # Subplot 4: Source Retrieval Breakdown (stacked bars)
    ax4 = fig.add_subplot(gs[1, 1])
    doc1_matched = []
    doc2_matched = []
    no_match = []
    
    for model in models:
        doc1_count = sum(1 for s in rag_scores[model] 
                        if s.get("retrieval_metrics", {}).get("found_relevant_doc_1", False))
        doc2_count = sum(1 for s in rag_scores[model] 
                        if s.get("retrieval_metrics", {}).get("found_relevant_doc_2", False))
        total = len(rag_scores[model])
        no_match_count = total - doc1_count - doc2_count
        
        doc1_matched.append(doc1_count)
        doc2_matched.append(doc2_count)
        no_match.append(no_match_count)
    
    x = np.arange(len(models))
    width = 0.6
    
    bars1 = ax4.bar(x, doc1_matched, width, label='Matched Doc 1', 
                   color='#3498db', edgecolor='black', linewidth=2, alpha=0.9)
    bars2 = ax4.bar(x, doc2_matched, width, bottom=doc1_matched, label='Matched Doc 2', 
                   color='#2ecc71', edgecolor='black', linewidth=2, alpha=0.9)
    bars3 = ax4.bar(x, no_match, width, bottom=np.array(doc1_matched) + np.array(doc2_matched), 
                   label='No Match', color='#95a5a6', edgecolor='black', linewidth=2, alpha=0.8)
    
    ax4.set_ylabel('Number of Questions', fontsize=13, fontweight='bold')
    ax4.set_title('Source Retrieval Breakdown', fontsize=15, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=12, framealpha=0.95)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add labels on stacked bars
    for i in range(len(models)):
        if doc1_matched[i] > 0:
            ax4.text(i, doc1_matched[i]/2, f'{doc1_matched[i]}', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        if doc2_matched[i] > 0:
            ax4.text(i, doc1_matched[i] + doc2_matched[i]/2, f'{doc2_matched[i]}', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if no_match[i] > 0:
            ax4.text(i, doc1_matched[i] + doc2_matched[i] + no_match[i]/2, f'{no_match[i]}', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.savefig(output_dir / 'rag_improvement_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated rag_improvement_analysis.png")

def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    test_set, baseline_scores, rag_scores, question_map = load_data()
    
    print("\nGenerating improved plots...")
    plot_performance_comparison(baseline_scores, rag_scores, output_dir)
    plot_language_analysis(test_set, baseline_scores, rag_scores, question_map, output_dir)
    plot_rag_improvement_analysis(baseline_scores, rag_scores, output_dir)
    
    print("\n✓ All improved plots generated successfully!")

if __name__ == "__main__":
    main()

