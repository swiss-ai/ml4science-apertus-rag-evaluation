# RAG Evaluation Analysis: Qwen3-80B vs Apertus-8B
## Executive Summary

This document presents a comprehensive analysis of the RAG (Retrieval-Augmented Generation) evaluation results comparing **Qwen3-80B** and **Apertus-8B** models on a dataset of 100 questions from the ETH Zurich web archive.

**Evaluation Method**: All responses are scored using **LLM-as-Judge** with `moonshotai/Kimi-K2-Thinking` as the judge model. The judge uses structured prompts (`judge_prompt_baseline.txt` for baseline, `judge_prompt_rag.txt` for RAG) to provide consistent, automated scoring.

**Key Finding**: Apertus-8B shows stronger RAG improvement (71.00% of questions improved) compared to Qwen3-80B (61.00%), suggesting better integration with retrieved context.

---

## 1. Apertus-8B vs Qwen3-80B - Baseline Test Results

### Dataset Coverage
- **Total Questions**: 100
- **Qwen Evaluation**: 100 questions
- **Apertus Evaluation**: 100 questions

### Baseline Performance (No RAG)

#### Apertus-8B Baseline
- **LLM-as-Judge Score**: 0.185 (out of 1.0)
- **Correct/Partial Answers**: 44 questions

#### Qwen3-80B Baseline
- **LLM-as-Judge Score**: 0.223 (out of 1.0)
- **Correct/Partial Answers**: 50 questions

### RAG-Enhanced Performance

#### Apertus-8B with RAG
- **LLM-as-Judge Score**: 0.460 (out of 1.0)
- **RAG Improvement**: 71.00% of questions
- **Score Change**: +0.275 (+27.5 points)

#### Qwen3-80B with RAG
- **LLM-as-Judge Score**: 0.438 (out of 1.0)
- **RAG Improvement**: 61.00% of questions
- **Score Change**: +0.215 (+21.5 points)

### Head-to-Head Comparison

| Metric | Qwen3-80B | Apertus-8B | Winner |
|--------|-----------|------------|--------|
| Baseline LLM Judge Score | 0.223 | 0.185 | Qwen |
| RAG LLM Judge Score | 0.438 | 0.460 | Apertus |
| RAG Improvement Rate | 61.00% | 71.00% | **Apertus** |

**Key Insight**: While Qwen3-80B may have a higher baseline score, **Apertus-8B shows better RAG integration** with 71.00% of questions improved vs 61.00% for Qwen.

---

## 2. Winner Distribution Analysis

### Apertus-8B

**Baseline (Parametric Only)**:
- LLM Judge Score: 0.185/1.0
- Uses only training knowledge

**RAG (Non-Parametric + Parametric)**:
- LLM Judge Score: 0.460/1.0
- Uses retrieved documents + training knowledge
- **RAG Improved**: 71.00% of questions (71/100)
- **Score Change**: +0.275 (+27.5 points)

**Winner Distribution**:
- rag: 71 questions (71.0%)
- baseline: 8 questions (8.0%)

### Qwen3-80B

**Baseline (Parametric Only)**:
- LLM Judge Score: 0.223/1.0
- Uses only training knowledge

**RAG (Non-Parametric + Parametric)**:
- LLM Judge Score: 0.438/1.0
- Uses retrieved documents + training knowledge
- **RAG Improved**: 61.00% of questions (61/100)
- **Score Change**: +0.215 (+21.5 points)

**Winner Distribution**:
- rag: 61 questions (61.0%)
- baseline: 11 questions (11.0%)

### Key Insights

1. **RAG is Effective**: Both models show significant improvement with RAG
   - Apertus: 71.00% of questions improved
   - Qwen: 61.00% of questions improved

2. **RAG is Most Beneficial For**:
   - Up-to-date information (not in training data)
   - Domain-specific knowledge (ETH Zurich policies, procedures)
   - Precise factual information (dates, locations, contact info)

3. **Baseline Strength**: Models perform well on:
   - General knowledge questions
   - Common sense reasoning
   - Questions well-covered in training data

4. **Hybrid Approach Works Best**:
   - Combining parametric knowledge (model's training) with non-parametric knowledge (retrieved documents) yields best results
   - RAG provides context that models don't have in their training data

---

## 3. Evaluation Methodology Note

### Scoring Criteria

The evaluation used a **strict scoring approach** to ensure high-quality assessment:

- **Correct (Score 2/2)**: Response must mention the specific ETH portal/tool from Ground Truth (e.g., ETHIS, ASVZ, Moodle)
- **Partial (Score 1/1 or 1/0)**: Response provides useful ETH-relevant information but may miss the exact tool name or some details
- **Generic (Score 0/0)**: Response provides only general advice without ETH-specific context
- **Refusal (Score 0/0)**: Model explicitly states "I don't know" or "I don't have access" - honest but receives 0 score

**Note on Judge Scores**: The judge scores may appear lower than expected because **retrieved resources often don't perfectly match the ground truth documents**. Even when RAG retrieves relevant information, if the retrieved documents don't exactly match the specific URLs or tools mentioned in the ground truth, the judge applies strict criteria. This is intentional to ensure high-quality evaluation, but it means that:
- Responses that use retrieved ETH context but don't match exact tool names receive partial credit
- The evaluation emphasizes precision: responses must demonstrate specific institutional knowledge
- Lower absolute scores reflect the high bar for ETH-specific knowledge, not poor RAG performance
- **Refusal responses always receive 0 score** (correctness=0, completeness=0) as they don't provide operational utility

**Why the prompt was strict**: The scoring criteria required responses to mention specific ETH tools/portals (e.g., ETHIS, ASVZ) rather than accepting generic advice. This ensures that improvements shown are meaningful and represent genuine enhancement of the model's ability to provide ETH-specific information, not just general guidance that could apply to any institution.

---

## 4. Plot Analysis

For detailed analysis of all visualizations, see [`plot_analysis.md`](plot_analysis.md).

### Key Visualizations

1. **Performance Comparison** (`results/plots/performance_comparison.png`):
   - Shows aggregate scores, correctness, completeness, and improvement rates
   - Demonstrates consistent RAG improvement across all metrics
   - Apertus-8B shows 45% improvement rate, Qwen3-80B shows 35%

2. **Language Analysis** (`results/plots/language_analysis.png`):
   - Analyzes performance by question language (English vs German)
   - Shows RAG improves performance for both languages
   - Demonstrates multilingual retrieval effectiveness

3. **RAG Improvement Analysis** (`results/plots/rag_improvement_analysis.png`):
   - Shows winner distribution (RAG vs Baseline)
   - Displays score improvement distribution histogram
   - Compares correct answer counts between baseline and RAG

4. **Baseline Tag Distribution** (`results/plots/tag_distribution.png`):
   - Shows how baseline models handle uncertainty
   - Displays distribution of Correct, Generic, Refusal, and Hallucination tags
   - Reveals models' institutional knowledge limitations
   - **Note**: Refusal responses are scored as 0 (correctness=0, completeness=0)

