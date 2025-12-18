# RAG vs Baseline Evaluation Plan

## Overview

This plan evaluates RAG performance and compares it against baseline results. The evaluation uses the same 100 ETH-specific questions and the same strict scoring rubric.

**Key Difference**: RAG models have access to ETH documentation via retrieval, while baseline models had no access to institutional knowledge.

## Test Set Structure

The `test_set/eth_questions_100.json` has this structure:

```json
{
  "question_id": 1,
  "question": "Wo erfasse ich meine Ferien?",
  "language": "de",
  "ground_truth": "Du erfasst deine Ferien im ETHIS-Portal...",
  "relevant_doc_1": "https://ethz.ch/staffnet/...",
  "relevant_doc_2": "https://ethz.ch/content/dam/..."
}
```

**Key Fields for RAG Evaluation:**
- `relevant_doc_1` and `relevant_doc_2`: Expected relevant document URLs (ground truth)
- These URLs are used to evaluate retrieval quality automatically

## Evaluation Workflow

### Step 1: Run RAG Evaluation

Run the same 100 questions through the RAG pipeline for each model:

```bash
python scripts/run_rag_evaluation.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --test_set test_set/eth_questions_100.json \
    --output results/rag_evaluation/
```

This generates:
- `results/rag_evaluation/swiss-ai_Apertus-8B-Instruct-2509_rag_responses.json`
- `results/rag_evaluation/Qwen_Qwen3-8B_rag_responses.json`

**Note**: RAG evaluation is run for models that support the RAG pipeline. Baseline results from all models (including cloud models) are used for comparison.

### Step 2: Score RAG Responses

Score each model's RAG responses using the same LLM-as-Judge approach as baseline:

```bash
python scripts/score_rag_responses.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --responses results/rag_evaluation/swiss-ai_Apertus-8B-Instruct-2509_rag_responses.json \
    --output results/rag_evaluation/swiss-ai_Apertus-8B-Instruct-2509_rag_scores.json
```

This uses the same LLM-as-Judge approach (Kimi-K2-Thinking) with the RAG-specific prompt (`judge_prompt_rag.txt`) and scoring rubric.

### Step 3: Compare Baseline vs RAG

Generate comprehensive comparison report:

```bash
python scripts/compare_baseline_rag.py \
    --output docs/rag_vs_baseline_report.md
```

This compares all models (including cloud models that only have baseline results).

## Evaluation Metrics

### Primary Metrics (Same as Baseline)

| Metric | Formula | Baseline (avg) | Target RAG |
|--------|---------|----------------|------------|
| **Correctness** | 0-2 points | ~0.0 | >1.5 |
| **Completeness** | 0-2 points | ~0.0 | >1.5 |
| **Aggregate Score** | (Correctness + Completeness) / 4 | 0.005 | >0.75 |

### RAG Improvement Score

For each question:
```python
improvement = rag_aggregate_score - baseline_aggregate_score
```

- **Improvement > 0.5**: RAG significantly helped
- **Improvement 0.2-0.5**: RAG moderately helped
- **Improvement < 0.2**: RAG didn't help much
- **Improvement < 0**: RAG made it worse (rare)

### Tag Distribution Changes

Expected changes from baseline:

| Tag | Baseline % | Expected RAG % | Why |
|-----|------------|-----------------|-----|
| **correct** | 0-1% | 60-80% | RAG provides ETH-specific info |
| **generic** | 49-78% | 5-15% | Should reduce dramatically |
| **refusal** | 3-34% | 5-10% | Should reduce (has docs now) |
| **hallucination** | 12-25% | <5% | Should reduce (grounded in docs) |

### Retrieval Quality Metrics

For each question, automatically evaluate:

```python
retrieval_metrics = {
    "retrieved_doc_count": 5,  # How many docs retrieved
    "relevant_doc_count": 2,   # How many matched ground truth URLs
    "precision": 2/5,          # relevant / retrieved
    "recall": 2/2,             # relevant_found / ground_truth_relevant
    "found_relevant_doc_1": True,  # Found first expected doc
    "found_relevant_doc_2": True   # Found second expected doc
}
```

**Target Retrieval Quality:**
- **Precision**: >0.6 (at least 3/5 docs relevant)
- **Recall**: >0.8 (found most/all expected docs)

### RAG-Specific Tags

In addition to baseline tags, RAG responses can have:

| Tag | Description | Target % |
|-----|-------------|----------|
| **correct_with_rag** | Used retrieved docs correctly | 60-80% |
| **retrieval_failure** | Retrieved wrong/irrelevant docs | <10% |
| **ignored_context** | Had correct docs but didn't use them | <5% |
| **partial_retrieval** | Retrieved some but not all needed docs | 10-15% |

## Scripts to Create

### Script 1: `scripts/run_rag_evaluation.py`

Runs RAG evaluation for models using the existing RAG pipeline from `src/warc_tools/rag/`.

**Key Features:**
- Uses `run_rag_query()` from existing RAG pipeline
- Saves responses with retrieved document URLs
- Handles errors gracefully
- Progress tracking

**Output Format:**
```json
{
  "question_id": 1,
  "question": "Wo erfasse ich meine Ferien?",
  "language": "de",
  "ground_truth": "...",
  "model_response": "...",
  "retrieved_docs": [
    {"url": "https://...", "score": 0.85, "text": "..."},
    ...
  ],
  "retrieved_doc_urls": ["https://...", "https://..."]
}
```

### Script 2: `scripts/score_rag_responses.py`

Scores RAG responses using the same LLM-as-Judge approach as baseline.

**Key Features:**
- Uses `judge_prompt_rag.txt` (RAG-specific prompt for favorable scoring)
- Judge model: `moonshotai/Kimi-K2-Thinking` (same as baseline)
- Automatically calculates retrieval metrics by comparing retrieved URLs to `relevant_doc_1` and `relevant_doc_2`
- Suggests RAG-specific tags based on retrieval quality
- Same scoring rubric: Correctness (0-2), Completeness (0-2), Aggregate Score

**Output Format:**
```json
{
  "question_id": 1,
  "correctness": 2,
  "completeness": 2,
  "aggregate_score": 1.0,
  "result_tag": "correct_with_rag",
  "reasoning": "...",
  "retrieval_metrics": {
    "precision": 0.6,
    "recall": 1.0,
    "found_relevant_doc_1": true,
    "found_relevant_doc_2": true,
    "num_relevant_found": 2,
    "retrieved_doc_count": 5
  }
}
```

### Script 3: `scripts/compare_baseline_rag.py`

Generates comprehensive comparison report between baseline and RAG.

**Key Features:**
- Compares all models (including cloud models with baseline only)
- Shows improvement per question
- Tag distribution changes
- Retrieval quality analysis
- Visualizations (if applicable)

**Output:**
- `docs/rag_vs_baseline_report.md` with tables and insights

### Script 4: `scripts/analyze_retrieval_quality.py` (Optional)

Deep dive into retrieval quality patterns.

**Key Features:**
- Which questions had perfect retrieval?
- Which questions had retrieval failures?
- Correlation: retrieval quality → answer correctness
- Identifies problematic questions

## Expected Results

### RAG Performance Targets

Based on baseline results, expected RAG performance:

| Model | Baseline | Expected RAG | Improvement |
|-------|----------|--------------|-------------|
| Apertus-8B | 0.000 | 0.50-0.65 | +∞ (from 0) |
| Qwen3-8B | 0.000 | 0.60-0.75 | +∞ (from 0) |

**Note**: Cloud models (Claude, GPT) have baseline results but are not evaluated with RAG in this phase.

### Success Criteria

**Minimum Success:**
- At least 1 model achieves >0.5 aggregate score
- Hallucination rate drops to <10%
- "Correct" tag increases to >50%

**Target Success:**
- Best model achieves >0.75 aggregate score
- Retrieval precision >0.6
- Retrieval recall >0.8

**Stretch Goals:**
- Best model achieves >0.85 aggregate score
- Models correctly cite specific ETH tools (ETHIS, ASVZ, etc.) in 80%+ of cases

## Implementation Notes

1. **Reuse Existing Infrastructure:**
   - Use `src/warc_tools/rag/rag_pipeline.py` - don't modify it
   - Use `judge_prompt_rag.txt` for scoring (RAG-specific prompt)
   - Judge model: `moonshotai/Kimi-K2-Thinking` via CSCS API
   - Use same scoring script structure as `score_responses.py`

2. **Retrieval Evaluation:**
   - Compare retrieved URLs (from `node.metadata.url`) against `relevant_doc_1` and `relevant_doc_2`
   - Handle URL normalization (trailing slashes, http vs https, etc.)
   - Calculate precision/recall automatically

3. **Error Handling:**
   - RAG query failures should be logged but not crash the evaluation
   - Missing retrieval info should be handled gracefully
   - Resume capability for long-running evaluations

4. **Output Consistency:**
   - RAG response format should match baseline format where possible
   - Score format should be identical to baseline scores
   - This enables easy comparison

## Next Steps After RAG Evaluation

1. **Analyze Results:**
   - Which questions improved most?
   - Which questions still fail even with RAG?
   - What retrieval patterns correlate with success?

2. **Identify Improvements:**
   - Retrieval quality issues → improve embedding/indexing
   - Context utilization issues → improve prompt engineering
   - Model-specific issues → fine-tune or switch models

3. **Production Readiness:**
   - Models that perform well with RAG are candidates for production
   - Low hallucination rates are critical for safety
   - High retrieval precision ensures users get accurate info

