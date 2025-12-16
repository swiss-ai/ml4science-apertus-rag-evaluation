# LLM Evaluation Best Practices

This document outlines best practices for evaluating Large Language Models (LLMs) in question-answering tasks, specifically for comparing RAG (Retrieval-Augmented Generation) systems against baseline models.

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Evaluation Methods](#evaluation-methods)
3. [Data Preparation](#data-preparation)
4. [Running Evaluations](#running-evaluations)
5. [Interpreting Results](#interpreting-results)
6. [Common Pitfalls](#common-pitfalls)
7. [Best Practices Summary](#best-practices-summary)

---

## Evaluation Overview

### What We're Evaluating

When comparing RAG vs. baseline models, we evaluate:

1. **Answer Quality**: How well does the model answer the question?
2. **Factual Accuracy**: Are the facts in the answer correct?
3. **Completeness**: Does the answer cover all important aspects?
4. **Relevance**: Are retrieved documents relevant to the question?

### Evaluation Setup

- **Baseline (without RAG)**: Model answers using only its parametric knowledge
- **RAG (with retrieval)**: Model answers using retrieved documents from Elasticsearch

---

## Evaluation Methods

We use three complementary evaluation methods:

### 1. Exact Match (EM)

**What it measures**: Whether the predicted answer exactly matches the reference answer.

**How it works**:
- Normalizes whitespace and case
- Compares strings character-by-character
- Returns: `True` (match) or `False` (no match)

**When to use**:
-  Good for factual questions with specific answers
-  Fast and deterministic
-  Too strict for questions with multiple valid phrasings
-  Doesn't account for semantic equivalence

**Example**:
```
Reference: "ETHIS-Portal"
Predicted: "ETHis Portal"
Result: Exact Match = False (case/whitespace difference)
```

### 2. Multiple Choice

**What it measures**: Whether the model selected the correct option (A, B, C, D, E).

**How it works**:
- Extracts choice letters from both reference and predicted answers
- Compares the letters
- Returns: `True` (correct), `False` (incorrect), or `None` (not applicable)

**When to use**:
-  Perfect for multiple-choice questions
-  Objective and unambiguous
-  Only applicable to multiple-choice formats

**Example**:
```
Reference: "Answer: B"
Predicted: "The correct answer is B"
Result: Multiple Choice = True
```

### 3. LLM-as-Judge

**What it measures**: Semantic correctness and completeness using another LLM as an evaluator.

**How it works**:
- Uses a separate LLM (judge model) to evaluate the predicted answer
- Judge considers: correctness, completeness, accuracy
- Returns: Score (0.0-1.0), Correct (bool), Reasoning (str)

**When to use**:
-  Handles semantic equivalence
-  Can evaluate complex, multi-part answers
-  Provides reasoning for decisions
-  Slower and more expensive
-  May have bias from judge model

**Example**:
```
Question: "Where do I record my vacation?"
Reference: "ETHIS-Portal"
Predicted: "You record vacation in the ETHIS system"
Result: LLM Judge Score = 0.85, Correct = True
Reasoning: "The answer is semantically correct and mentions the correct system, though slightly different phrasing."
```

### 4. LLM-as-Judge Comparison (Baseline vs RAG)

**What it measures**: Direct comparison between baseline (no RAG) and RAG answers to determine which performs better.

**How it works**:
- Uses an LLM judge to compare both answers side-by-side
- Judge evaluates both answers against the reference
- Determines which answer is better overall
- Returns: Winner ("baseline", "rag", or "tie"), Scores for both, Reasoning, and whether RAG improved the answer

**When to use**:
-  Directly answers: "Did RAG help?"
-  Provides head-to-head comparison
-  Shows which answer is better and why
-  Most useful for understanding RAG impact

**Example**:
```
Question: "Where do I record my vacation?"
Reference: "ETHIS-Portal"
Baseline: "You can record vacation in your company's HR system"
RAG: "You record vacation in the ETHIS-Portal (ETH Information and Support System)"
Result: 
  Winner: "rag"
  Baseline Score: 0.65
  RAG Score: 0.92
  RAG Improved: True
  Reasoning: "The RAG answer is more specific, mentions the exact system name, and provides additional context."
```

---

## Data Preparation

### Input Excel File Format

Your evaluation dataset should be an Excel file (`.xlsx`) with at least these columns:

| Column | Required | Description |
|--------|----------|------------|
| `question` | ✅ Yes | The question to be answered |
| `answer` | ✅ Yes | Ground truth (golden) answer |
| `relevant_doc_1` | ✅ Yes | First relevant document URL (for reference) |
| `relevant_doc_2` | ✅ Yes | Second relevant document URL (for reference) |
| `baseline_answer` | ⚠️ Optional | Pre-computed baseline answer (if already run) |
| `rag_answer` | ⚠️ Optional | Pre-computed RAG answer (if already run) |

### Column Naming Convention

After evaluation, the script will create these columns:

- `golden_answer` (alias for `answer`)
- `qwen_answer_wo_rag` (alias for `baseline_answer`)
- `qwen_answer_with_rag` (alias for `rag_answer`)

### Data Quality Checklist

Before running evaluation:

- [ ] All questions are non-empty
- [ ] All reference answers are provided
- [ ] Relevant documents are valid URLs
- [ ] Questions are clear and unambiguous
- [ ] Reference answers are accurate and complete
- [ ] Dataset size is appropriate (100+ questions recommended)

---

## Running Evaluations

### Step 1: Prepare Environment

Set up your `.env` file with:

```bash
# For RAG evaluation
ES_URL=http://localhost:9200
ES_INDEX_NAME=warc_rag_index
LLM_PROVIDER=cscs
LLM_MODEL=Qwen/Qwen3-8B
LLM_BASE_URL=https://api.swissai.cscs.ch/v1
LLM_API_KEY=<your-key>

# For embeddings
EMBED_PROVIDER=cscs
EMBED_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0
EMBED_BASE_URL=https://api.swissai.cscs.ch/v1
EMBED_API_KEY=<your-key>

# For LLM-as-judge (optional, uses LLM_* if not set)
JUDGE_LLM_MODEL=Apertus-8B  # or use same as LLM_MODEL
```

### Step 2: Run Baseline Evaluation (if not done)

If you don't have `baseline_answer` column yet:

```bash
python -m warc_tools.baseline.cli \
    evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-qwen3-baseline.xlsx \
    Qwen/Qwen3-8B
```

### Step 3: Run RAG Evaluation

```bash
# Option 1: Use the evaluation script (runs RAG automatically)
python scripts/evaluate_answers.py \
    evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-results.xlsx \
    --run-rag

# Option 2: Run RAG separately first
python -m warc_tools.rag.cli \
    --eval-xlsx evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-qwen3-rag.xlsx
```

### Step 4: Run Comprehensive Evaluation

```bash
python scripts/evaluate_answers.py \
    evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-results.xlsx \
    --run-rag \
    --judge-model Apertus-8B
```

**Options**:
- `--run-rag`: Run RAG evaluation if `rag_answer` column is missing
- `--skip-llm-judge`: Skip LLM-as-judge (faster, but less comprehensive)
- `--judge-model MODEL`: Use specific model for judging (default: same as `LLM_MODEL`)

### Step 5: Review Results

The output Excel file will contain:

1. **Original columns**: All input columns preserved
2. **Renamed columns**: `golden_answer`, `qwen_answer_wo_rag`, `qwen_answer_with_rag`
3. **Evaluation metrics**:
   - `exact_match_baseline`, `exact_match_rag`
   - `multiple_choice_baseline`, `multiple_choice_rag`
   - `llm_judge_score_baseline`, `llm_judge_score_rag`
   - `llm_judge_correct_baseline`, `llm_judge_correct_rag`
   - `llm_judge_reasoning_baseline`, `llm_judge_reasoning_rag`
   - `llm_compare_winner` (which answer won: "baseline", "rag", or "tie")
   - `llm_compare_baseline_score` (score for baseline answer)
   - `llm_compare_rag_score` (score for RAG answer)
   - `llm_compare_rag_improved` (whether RAG improved the answer)
   - `llm_compare_reasoning` (explanation of the comparison)

---

## Interpreting Results

### Exact Match Scores

- **High EM (>80%)**: Answers are very precise and match reference closely
- **Low EM (<50%)**: Answers may be semantically correct but phrased differently
- **Action**: If EM is low but answers seem correct, use LLM-as-judge

### Multiple Choice Scores

- **100%**: Perfect on multiple-choice questions
- **<80%**: Model struggles with multiple-choice format
- **Action**: Check if model is extracting choice letters correctly

### LLM-as-Judge Scores

- **Score 0.8-1.0**: Excellent answer quality
- **Score 0.6-0.8**: Good answer, minor issues
- **Score 0.4-0.6**: Acceptable but needs improvement
- **Score <0.4**: Poor answer quality

**Read the reasoning**: The `llm_judge_reasoning` column explains why the score was given.

### Comparing Baseline vs. RAG

Look for:

1. **RAG improves accuracy**: `exact_match_rag > exact_match_baseline`
2. **RAG improves completeness**: `llm_judge_score_rag > llm_judge_score_baseline`
3. **RAG provides sources**: Check `rag_source_urls` column
4. **Direct comparison**: Check `llm_compare_rag_improved` column
   - Shows percentage of questions where RAG improved answers
   - `llm_compare_winner` shows which answer won for each question
   - `llm_compare_reasoning` explains why one answer is better

---

## Common Pitfalls

### 1. Inconsistent Evaluation

❌ **Wrong**: Running evaluation on different subsets of questions
✅ **Right**: Use the same evaluation dataset for all models

### 2. Temperature Settings

❌ **Wrong**: Using `temperature > 0` for deterministic evaluation
✅ **Right**: Use `temperature=0.0` for reproducible results

### 3. Missing Reference Answers

❌ **Wrong**: Evaluating without ground truth answers
✅ **Right**: Always provide accurate reference answers

### 4. Ignoring Context

❌ **Wrong**: Evaluating answers without considering question context
✅ **Right**: Use LLM-as-judge to account for semantic equivalence

### 5. Over-relying on Exact Match

❌ **Wrong**: Only using exact match for all questions
✅ **Right**: Combine multiple metrics (EM + LLM-as-judge)

### 6. Not Checking Retrieved Documents

❌ **Wrong**: Only evaluating final answers
✅ **Right**: Verify that `rag_source_urls` contain relevant documents

---

## Best Practices Summary

### ✅ DO:

1. **Use multiple evaluation methods**: Combine Exact Match, Multiple Choice, and LLM-as-Judge
2. **Set temperature to 0.0**: For deterministic, reproducible results
3. **Use same dataset**: Evaluate all models on identical questions
4. **Document your setup**: Record model versions, dates, configurations
5. **Review individual answers**: Don't just look at aggregate scores
6. **Check retrieved documents**: Verify RAG is retrieving relevant sources
7. **Use appropriate judge model**: Use a strong model (e.g., Apertus-8B) for LLM-as-judge
8. **Handle errors gracefully**: Log failed queries for manual review

### ❌ DON'T:

1. **Don't mix evaluation datasets**: Use consistent questions across runs
2. **Don't use high temperature**: Keep `temperature=0.0` for evaluation
3. **Don't ignore semantic equivalence**: Use LLM-as-judge for nuanced evaluation
4. **Don't skip error handling**: Check for empty answers, API failures, etc.
5. **Don't evaluate without reference**: Always have ground truth answers
6. **Don't over-interpret small differences**: Consider statistical significance
7. **Don't forget to save configurations**: Record all environment variables and settings

---

## Example Workflow

```bash
# 1. Prepare evaluation dataset
# (Ensure evaluation/evaluation-qwen3.xlsx has: question, answer, relevant_doc_1, relevant_doc_2)

# 2. Run baseline evaluation (if needed)
python -m warc_tools.baseline.cli \
    evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-qwen3-baseline.xlsx \
    Qwen/Qwen3-8B

# 3. Run comprehensive evaluation
python scripts/evaluate_answers.py \
    evaluation/evaluation-qwen3.xlsx \
    evaluation/evaluation-results.xlsx \
    --run-rag \
    --judge-model Apertus-8B

# 4. Review results
# Open evaluation/evaluation-results.xlsx and check:
# - exact_match_rag vs exact_match_baseline
# - llm_judge_score_rag vs llm_judge_score_baseline
# - Individual answers in qwen_answer_wo_rag vs qwen_answer_with_rag
```

---

## Questions?

For questions on:
- **Evaluation methodology**: See this document
- **Script usage**: Run `python scripts/evaluate_answers.py --help`
- **Model availability**: Check CSCS/SwissAI API documentation
- **RAG setup**: See `EVALUATION_METHODOLOGY.md`

