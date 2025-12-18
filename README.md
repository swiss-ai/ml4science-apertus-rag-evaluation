# ML4Science Apertus RAG Evaluation

A comprehensive evaluation framework for comparing baseline LLM performance against Retrieval-Augmented Generation (RAG) systems on ETH Zurich-specific questions.

## Overview

This project evaluates various Large Language Models (LLMs) on their ability to answer institution-specific questions from the ETH Zurich web archive. The evaluation compares:

- **Baseline Performance**: Models without RAG (parametric knowledge only)
- **RAG Performance**: Models with access to retrieved ETH documents (non-parametric + parametric knowledge)

### Evaluated Models

- **Apertus-8B**: `swiss-ai/Apertus-8B-Instruct-2509` (self-hosted on CSCS)
- **Qwen3-8B**: `Qwen/Qwen3-8B` (self-hosted on CSCS)
- **Claude Sonnet 4.5**: Anthropic (manual collection via official interface)
- **GPT-5.2**: OpenAI (manual collection via official interface)

## Project Structure

```
ml4science-apertus-rag-evaluation/
├── docs/                          # Documentation
│   ├── evaluation_method.md       # Detailed evaluation methodology
│   ├── baseline_evaluation.md     # Baseline results summary
│   ├── rag_vs_baseline_report.md  # RAG vs Baseline comparison
│   ├── rag_evaluation_plan.md     # RAG evaluation methodology
│   └── plot_analysis.md            # Plot analysis and insights
├── scripts/                       # Evaluation scripts
│   ├── run_evaluation.py          # Run baseline evaluation
│   ├── score_responses.py         # Score baseline responses (LLM-as-Judge)
│   ├── run_rag_evaluation.py      # Run RAG evaluation
│   ├── score_rag_responses.py     # Score RAG responses (LLM-as-Judge)
│   ├── compare_models.py          # Compare baseline models
│   ├── compare_baseline_rag.py   # Compare baseline vs RAG
│   └── generate_improved_plots.py # Generate visualization plots
├── prompts/                       # Judge prompts
│   ├── judge_prompt_baseline.txt  # Prompt for baseline scoring
│   └── judge_prompt_rag.txt       # Prompt for RAG scoring
├── test_set/                      # Test dataset
│   ├── eth_questions_100.json     # 100 ETH questions (JSON)
│   └── eth_questions_100.xlsx     # 100 ETH questions (Excel)
├── results/                       # Evaluation results
│   ├── baseline_evaluation/      # Baseline responses and scores
│   ├── rag_evaluation/            # RAG responses and scores
│   └── plots/                     # Generated visualization plots
└── src/                           # Source code (RAG pipeline, etc.)
    └── warc_tools/
        ├── rag/                   # RAG pipeline implementation
        ├── extractor/              # WARC extraction tools
        ├── indexer/                # Elasticsearch indexing
        └── baseline/               # Baseline tools
```

## Important: Source Code vs Evaluation Results

**Note**: The evaluation framework does **not** directly use the source code in `src/warc_tools/baseline/` and `src/warc_tools/rag/`. 

- **`src/warc_tools/baseline/`**: Library code for baseline LLM calls (not used directly in evaluation)
- **`scripts/run_evaluation.py` and `results/baseline_evaluation/`**: Evaluation scripts and results used for baseline evaluation
- **`src/warc_tools/rag/`**: Library code for RAG pipeline (not used directly in evaluation)
- **`scripts/run_rag_evaluation.py` and `results/rag_evaluation/`**: Evaluation scripts and results used for RAG evaluation

## Evaluation Methodology

### LLM-as-Judge Approach

All responses are automatically scored using **LLM-as-Judge** with:
- **Judge Model**: `moonshotai/Kimi-K2-Thinking`
- **API**: CSCS OpenAI-compatible endpoint
- **Temperature**: 0 (deterministic scoring)
- **Prompts**: Structured prompts with clear scoring criteria

### Metrics

1. **Factual Correctness** (0-2 points): Accuracy of the information provided
2. **Completeness** (0-2 points): How fully the question is answered
3. **Aggregate Score**: `(Correctness + Completeness) / 4` (0-1 scale)
4. **Result Tags**: Correct, Partial, Generic, Refusal, Hallucination

### Reproducibility Settings

- **Temperature**: 0
- **Top_p**: 1
- **Max tokens**: 500 (baseline), 2000 (judge)
- **Random seed**: 42

## Quick Start

### Prerequisites

- Python 3.8+
- Access to CSCS cluster (for self-hosted models)
- API keys for cloud models (if evaluating)
- Environment variables configured (see `env.example`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml4science-apertus-rag-evaluation
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # If available
# Or install manually: openai, anthropic, google-generativeai, etc.
```

4. Configure environment variables:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

### Running Baseline Evaluation

1. **Run evaluation for a model**:
```bash
python scripts/run_evaluation.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --output results/baseline_evaluation/
```

2. **Score responses using LLM-as-Judge**:
```bash
python scripts/score_responses.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --input results/baseline_evaluation/swiss-ai_Apertus-8B-Instruct-2509_responses.json
```

3. **Compare all baseline models**:
```bash
python scripts/compare_models.py
```

### Running RAG Evaluation

1. **Run RAG evaluation**:
```bash
python scripts/run_rag_evaluation.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --test_set test_set/eth_questions_100.json \
    --output results/rag_evaluation/
```

2. **Score RAG responses**:
```bash
python scripts/score_rag_responses.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --responses results/rag_evaluation/swiss-ai_Apertus-8B-Instruct-2509_rag_responses.json
```

3. **Compare baseline vs RAG**:
```bash
python scripts/compare_baseline_rag.py
```

### Generating Plots

```bash
python scripts/generate_improved_plots.py
```

Plots are saved to `results/plots/`:
- `performance_comparison.png`: Multi-panel comparison
- `language_analysis.png`: Performance by language (DE/EN)
- `rag_improvement_analysis.png`: RAG improvement analysis
- `tag_distribution.png`: Result tag distribution

## Results

### Baseline Results

Baseline evaluation results are available in:
- `docs/baseline_evaluation.md`: Summary statistics and insights
- `results/baseline_evaluation/`: Individual model scores

**Key Findings**:
- Models show high reasoning ability but are "institutionally blind" without RAG
- Claude/GPT show high refusal rates (honest uncertainty)
- Qwen/Apertus attempt to answer but often provide generic advice

### RAG vs Baseline Comparison

Comprehensive comparison available in:
- `docs/rag_vs_baseline_report.md`: Detailed analysis
- `results/plots/`: Visualization plots

**Key Findings**:
- RAG significantly improves performance (60-70% of questions improved)
- Apertus-8B shows better RAG integration than Qwen3-8B
- Retrieved documents enable models to provide ETH-specific information

## Documentation

- **[Evaluation Methodology](docs/evaluation_method.md)**: Detailed methodology, metrics, and protocols
- **[Baseline Results](docs/baseline_evaluation.md)**: Baseline evaluation summary
- **[RAG Evaluation Plan](docs/rag_evaluation_plan.md)**: RAG evaluation methodology
- **[RAG vs Baseline Report](docs/rag_vs_baseline_report.md)**: Comprehensive comparison analysis
- **[Plot Analysis](docs/plot_analysis.md)**: Analysis of generated visualizations

## Environment Variables

Required environment variables (see `env.example`):

```bash
# CSCS API (for self-hosted models and judge)
CSCS_API_KEY=your_cscs_api_key
CSCS_BASE_URL=https://api.swissai.cscs.ch/v1

# Judge model (optional, defaults to Kimi-K2-Thinking)
JUDGE_API_KEY=your_judge_api_key
JUDGE_BASE_URL=https://api.swissai.cscs.ch/v1

# Cloud model APIs (optional, for Claude/GPT evaluation)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# RAG pipeline
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.swissai.cscs.ch/v1
```

## Notes

- **Cloud Models**: Claude and GPT responses were collected manually via official model interfaces (web search/tools disabled)
- **Judge Model**: All scoring uses `moonshotai/Kimi-K2-Thinking` via CSCS API
- **Reproducibility**: All evaluations use temperature=0 for deterministic results
- **RAG Pipeline**: Uses existing RAG implementation from `src/warc_tools/rag/`

## License

[Add your license information here]

## Citation

If you use this evaluation framework, please cite:

```bibtex
[Add citation information]
```

## Contact

[Add contact information]
