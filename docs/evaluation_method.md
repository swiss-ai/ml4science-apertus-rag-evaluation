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

## LLM-as-Judge Protocol

We use an automated LLM-as-Judge approach for consistent, scalable evaluation:

### Judge Model
- **Model**: `moonshotai/Kimi-K2-Thinking`
- **API**: CSCS OpenAI-compatible endpoint (`https://api.swissai.cscs.ch/v1`)
- **Temperature**: 0 (deterministic scoring)
- **Max Tokens**: 2000 (to capture full reasoning)

### Evaluation Process
1. **Load ground truth**: Read the expected answer from the test set JSON
2. **Load model response**: Read the model's response from the results JSON
3. **Send to judge**: The judge model evaluates each response using a structured prompt
4. **Extract scores**: Judge returns JSON with:
   - `correctness` (0-2 points): Factual accuracy
   - `completeness` (0-2 points): How fully the question is answered
   - `result_tag`: One of "Correct", "Partial", "Generic", "Refusal", "Hallucination"
   - `reasoning`: Explanation of the scoring decision
5. **Calculate aggregate**: `aggregate_score = (correctness + completeness) / 4`

### Judge Prompts
- **Baseline evaluation**: `prompts/judge_prompt_baseline.txt` (lenient scoring for models without RAG)
- **RAG evaluation**: `prompts/judge_prompt_rag.txt` (favorable scoring for models with retrieved context)

The judge prompts provide detailed scoring criteria and guidelines for consistent evaluation across all models.

## Evaluation Process

### Phase 1: Data Preparation
- Convert Excel file to JSON format (`data/test_set/eth_questions_100.json`)
- Validate all questions have ground truth answers
- Check language distribution (German vs English)

### Phase 2: Model Testing
For each model:
1. Run evaluation script: `python scripts/run_evaluation.py --model <model_name>`
2. Save raw responses to `results/{model_name}_responses.json`
3. Review responses for any obvious errors

### Phase 3: Automated Scoring (LLM-as-Judge)
1. Run scoring script: `python scripts/score_responses.py --model <model_name>`
2. Script automatically sends each response to Kimi-K2-Thinking judge model
3. Judge evaluates and returns scores for each question
4. Save scores to `results/baseline_evaluation/{model_name}_scores.json`

### Phase 4: Analysis
1. Run comparison script: `python scripts/compare_models.py`
2. Review comparison report: `results/model_comparison_report.md`
3. Generate visualizations and summary statistics

## Models to Evaluate

### Self-Hosted Models (CSCS)
- **Apertus-8B**: `swiss-ai/Apertus-8B-Instruct-2509`
- **Qwen3-8B**: `Qwen/Qwen3-8B`
- **Mistral-7B**: `mistralai/Mistral-7B-v0.1` (run on CSCS cluster)
- **Llama-8B**: `meta-llama/Llama-3.1-8B-Instruct` (latest version, run on CSCS cluster)

**Note**: The comparison focuses on similar-sized models: Mistral-7B vs Llama-8B vs Apertus-8B. All models must be launched on the CSCS cluster using `model-launch` before running evaluations.

**Launch Instructions for Llama-3.1-8B-Instruct:**
When launching via `model-launch`, use:
- `--model-path meta-llama/Llama-3.1-8B-Instruct`
- `--served-model-name meta-llama/Llama-3.1-8B-Instruct`

The model will be downloaded automatically if it doesn't exist at the full file path.

### Cloud Models
- **Claude Sonnet 4.5**: Responses collected manually via official Anthropic interface (web search and tools disabled)
- **GPT-5.2**: Responses collected manually via official OpenAI interface (web search and function calling disabled)
- **Gemini 2.0**: Google API (grounding/search disabled)

**Note**: Claude and GPT responses were obtained through manual interaction with the official model interfaces, ensuring that web search, tools, and grounding features were disabled to maintain fair baseline comparison without external knowledge retrieval.

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
- **LLM-as-Judge** (Kimi-K2-Thinking) provides automated, consistent scoring
- Judge model uses structured prompts with clear scoring criteria
- All scores are saved with full reasoning for transparency and review

