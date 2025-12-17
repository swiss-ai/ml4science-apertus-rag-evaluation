# Baseline Evaluation: Model Comparison Report

This report compares baseline performance (without RAG) across all evaluated models.

## Summary Statistics

| Model | Avg Correctness | Avg Completeness | Avg Aggregate | Perfect (2/2) | Zero (0/0) |
|-------|----------------|------------------|---------------|--------------|-----------|
| Qwen/Qwen3-Next-80B-A3B-Instruct | 0.0/2.0 | 0.0/2.0 | 0.0/1.0 | 0 (0.0%) | 100 (100.0%) |
| claude-sonnet-4.5 | 0.01/2.0 | 0.01/2.0 | 0.005/1.0 | 0 (0.0%) | 99 (99.0%) |
| gpt-5.2 | 0.01/2.0 | 0.01/2.0 | 0.005/1.0 | 0 (0.0%) | 99 (99.0%) |
| swiss-ai/Apertus-8B-Instruct-2509 | 0.0/2.0 | 0.0/2.0 | 0.0/1.0 | 0 (0.0%) | 100 (100.0%) |

## Result Tag Distribution

| Model | Correct | Generic | Refusal | Hallucination |
|-------|---------|---------|---------|---------------|
| Qwen/Qwen3-Next-80B-A3B-Instruct | 1 (1.0%) | 78 (78.0%) | 3 (3.0%) | 18 (18.0%) |
| claude-sonnet-4.5 | 0 (0.0%) | 49 (49.0%) | 26 (26.0%) | 25 (25.0%) |
| gpt-5.2 | 0 (0.0%) | 64 (64.0%) | 19 (19.0%) | 17 (17.0%) |
| swiss-ai/Apertus-8B-Instruct-2509 | 0 (0.0%) | 54 (54.0%) | 34 (34.0%) | 12 (12.0%) |

## Multilingual Performance

| Model | Language | Avg Correctness | Avg Completeness | Avg Aggregate |
|-------|----------|----------------|------------------|---------------|
| Qwen/Qwen3-Next-80B-A3B-Instruct | DE | 0.00 | 0.00 | 0.000 |
| Qwen/Qwen3-Next-80B-A3B-Instruct | EN | 0.00 | 0.00 | 0.000 |
| claude-sonnet-4.5 | DE | 0.03 | 0.03 | 0.015 |
| claude-sonnet-4.5 | EN | 0.00 | 0.00 | 0.000 |
| gpt-5.2 | DE | 0.03 | 0.03 | 0.015 |
| gpt-5.2 | EN | 0.00 | 0.00 | 0.000 |
| swiss-ai/Apertus-8B-Instruct-2509 | DE | 0.00 | 0.00 | 0.000 |
| swiss-ai/Apertus-8B-Instruct-2509 | EN | 0.00 | 0.00 | 0.000 |

## Key Insights

### Best Overall Performance
- **claude-sonnet-4.5**: 0.005 average aggregate score

### Most Honest (Highest Refusal Rate)
- **swiss-ai/Apertus-8B-Instruct-2509**: 34.0% honest refusals (admits when it doesn't know)

### Least Hallucination
- **swiss-ai/Apertus-8B-Instruct-2509**: 12.0% hallucination rate (invents false information)

## Visualizations

See the following chart in `results/plots/`:
- `tag_distribution.png`: How models handle uncertainty (Generic vs Refusal vs Hallucination)

## Why Are Baseline Scores 0%?

### 1. Parametric Blindness

LLMs only know what is on the public internet. They cannot "see" internal administrative details (like the name ETHIS-Portal or specific Euler quotas) because that data is locked in your WARC files and PDFs, not in their training sets.

**Example**: A model might know general Swiss employment law, but it doesn't know that ETH uses "ETHIS-Portal" specifically for vacation tracking. This institutional knowledge exists only in ETH's internal documentation.

### 2. The "Strict Auditor" Bar

Your scoring rubric is binary. In this test, **"Helpful" is not "Correct."** Providing general Swiss law or saying "check with HR" is a Generic failure because the model missed the specific institutional "needle" (e.g., ETHIS).

**Scoring Logic**:
- ✅ **Correct (Score 2)**: Must mention specific ETH tool (e.g., "ETHIS-Portal", "ASVZ", "Euler cluster")
- ❌ **Generic (Score 0)**: General advice like "use Excel" or "check your calendar" when Ground Truth specifies an internal ETH tool
- ❌ **Refusal (Score 0)**: Model honestly says "I don't know" (safer than hallucination, but still 0 for operational utility)
- ❌ **Hallucination (Score 0)**: Model invents fake ETH tool names or non-ETH URLs as if official

### 3. Intelligence vs. Knowledge

All models showed high reasoning ability, but they are **"institutionally blind."** Claude/Sonnet correctly admits it doesn't know (Safe Refusal), while Qwen tries to guess (Generic/Hallucination). Both result in zero operational utility for an ETH employee.

**What This Means**:
- Models can reason about general topics (Swiss law, university processes)
- Models cannot access institutional specifics (ETHIS, ASVZ, internal policies)
- **Without RAG**: Models either refuse (honest but useless) or guess (dangerous hallucinations)
- **With RAG**: Models can combine reasoning with retrieved ETH documents to provide correct answers

### What Makes a Model "Better" in This Baseline?

Since all models score near zero, the **important differentiators** are:

1. **Refusal Rate (Higher is Better)**:
   - Models that honestly say "I don't know" are safer than those that guess
   - Example: Apertus-8B has 34% refusal rate (most honest)
   - This indicates good self-awareness of knowledge limits

2. **Hallucination Rate (Lower is Better)**:
   - Models that invent fake ETH tools/URLs are dangerous for users
   - Example: Apertus-8B has 12% hallucination rate (safest)
   - Lower hallucination = better safety for production use

3. **Generic Advice Rate (Lower is Better)**:
   - Models giving generic advice ("use Excel", "check calendar") show they don't recognize the need for ETH-specific knowledge
   - Higher generic rate = less awareness of knowledge boundaries

### The Real Test: RAG Performance

**This baseline is not the final evaluation.** The real test is:
- How much do models **improve with RAG**?
- Which models best **utilize retrieved ETH documents**?
- Can models **combine general knowledge with ETH-specific context**?

The baseline scores establish the starting point. Models that:
- Have low hallucination rates (safe)
- Have high refusal rates (honest)
- Show awareness of knowledge limits

...are likely to perform better **with RAG** because they will correctly use the provided context instead of ignoring it or hallucinating.

## Interpretation

**Baseline Evaluation Context:**
- All models evaluated **without RAG** (no access to ETH-specific documents)
- Strict scoring: Must mention specific ETH tools (ETHIS, ASVZ, etc.) for score > 0
- Generic advice = Score 0 (failure for institutional knowledge test)
- Low scores are expected - models lack ETH-specific knowledge without RAG

**What to Look For:**
- **Refusal Rate**: Higher is better (honest uncertainty)
- **Hallucination Rate**: Lower is better (safety)
- **Generic Rate**: Lower is better (shows awareness of knowledge limits)
