# Baseline Evaluation Methodology

## Overview

This evaluation tests models on ETH-specific questions **WITHOUT RAG** to establish baseline performance. The evaluation uses an Excel file with 100 questions and ground truth answers.

## Metrics Definitions

### 1. Factual Correctness (0-2 points)
- **2 points**: Completely accurate answer with correct facts
- **1 point**: Partially correct, some inaccuracies but core information is right
- **0 points**: Wrong, contains false information, or completely incorrect

### 2. Completeness (0-2 points)
- **2 points**: Answers all parts of the question fully
- **1 point**: Partially answers the question, missing some aspects
- **0 points**: Does not answer the question or only provides irrelevant information

### 3. Hallucination Rate (%)
Percentage of responses that contain information not present in the ground truth or that contradicts known facts.

### 4. Refusal Rate (%)
Percentage of responses where the model explicitly says "I don't know", "I cannot answer", or refuses to provide an answer.

### 5. Multilingual Accuracy
Separate performance metrics for:
- **German questions** (lang="de")
- **English questions** (lang="en")

## Scoring Rubric

### Aggregate Score Calculation
For each question:
```
Aggregate Score = (Correctness + Completeness) / 4
```

- Maximum score per question: 1.0 (2+2)/4
- Minimum score per question: 0.0 (0+0)/4

### Overall Metrics
- **Average Score**: Mean of all aggregate scores
- **Accuracy**: Percentage of questions with aggregate score ≥ 0.75
- **Hallucination Rate**: (Number of hallucinated responses / Total responses) × 100
- **Refusal Rate**: (Number of refusals / Total responses) × 100

## Reproducibility Settings

All evaluations use consistent parameters for reproducibility:

- **Temperature**: 0 → Same question always gets same answer (reproducible)
- **Top_p**: 1 → No word filtering (standard setting)
- **Max tokens**: 500 → Responses capped at reasonable length
- **Random seed**: 42 (if applicable)

These settings ensure fair, consistent comparisons across all models.

## Manual Evaluation Protocol

1. **Load ground truth**: Read the expected answer from the Excel file (`answer` column)
2. **Load model response**: Read the model's response from the results JSON
3. **Compare independently**: Score each response without looking at other model scores
4. **Score Correctness**: Rate 0-2 based on factual accuracy
5. **Score Completeness**: Rate 0-2 based on how fully the question is answered
6. **Flag hallucinations**: Mark if the response contains false information
7. **Flag refusals**: Mark if the model explicitly refuses to answer
8. **Document ambiguities**: Note any cases where scoring is ambiguous

## Evaluation Process

### Phase 1: Data Preparation
- Convert Excel file to JSON format (`test_set/eth_questions_100.json`)
- Validate all questions have ground truth answers
- Check language distribution (German vs English)

### Phase 2: Model Testing
For each model:
1. Run evaluation script: `python scripts/run_evaluation.py --model <model_name>`
2. Save raw responses to `results/{model_name}_responses.json`
3. Review responses for any obvious errors

### Phase 3: Manual Scoring
1. Run scoring script: `python scripts/score_responses.py --model <model_name>`
2. Score each response interactively
3. Save scores to `results/{model_name}_scores.json`

### Phase 4: Analysis
1. Run comparison script: `python scripts/compare_models.py`
2. Review comparison report: `results/model_comparison_report.md`
3. Generate visualizations and summary statistics

## Models to Evaluate

### Self-Hosted Models (CSCS)
- **Apertus-8B**: `swiss-ai/Apertus-8B-Instruct-2509`
- **Qwen3-80B**: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Mistral-7B**: `mistralai/Mistral-7B-v0.1` (run on CSCS cluster)
- **Llama-8B**: `meta-llama/Llama-3.1-8B-Instruct` (latest version, run on CSCS cluster)

**Note**: The comparison focuses on similar-sized models: Mistral-7B vs Llama-8B vs Apertus-8B. All models must be launched on the CSCS cluster using `model-launch` before running evaluations.

**Launch Instructions for Llama-3.1-8B-Instruct:**
When launching via `model-launch`, use:
- `--model-path meta-llama/Llama-3.1-8B-Instruct`
- `--served-model-name meta-llama/Llama-3.1-8B-Instruct`

The model will be downloaded automatically if it doesn't exist at the full file path.

### Cloud Models (API)
- **Claude Sonnet 4.5**: Anthropic API (tools disabled)
- **GPT-4**: OpenAI API (function calling disabled)
- **Gemini 2.0**: Google API (grounding/search disabled)

**CRITICAL**: For cloud models, ensure web search, tools, and grounding are **disabled** to ensure fair comparison.

## Output Files

### Raw Responses
- `results/{model_name}_responses.json`: Full model responses with metadata

### Scores
- `results/{model_name}_scores.json`: Manual scores for each question

### Comparison Report
- `results/model_comparison_report.md`: Comprehensive comparison of all models

### Visualizations
- `results/plots/`: Bar charts, category breakdowns, multilingual comparisons

## Notes

- All evaluations are done **without RAG** to establish true baseline performance
- Temperature is set to 0 for deterministic results
- Manual scoring ensures quality and consistency
- Ambiguous cases should be documented for review

