# Plot Analysis: RAG vs Baseline Evaluation

This document provides detailed analysis of the visualizations generated from the RAG vs Baseline evaluation.

## Overview

The evaluation compares baseline LLM performance (without RAG) against RAG-enhanced performance for two models:
- **Apertus-8B**: 8B parameter model
- **Qwen3-8B**: 8B parameter model

## Plot 1: Performance Comparison (`performance_comparison.png`)

### Description
This multi-panel visualization shows aggregate scores, correctness, completeness, and improvement rates for both baseline and RAG scenarios.

### Key Insights

1. **Aggregate Score Comparison**:
   - Baseline scores are significantly lower (0.18-0.22) compared to RAG scores (0.28-0.29)
   - This demonstrates that RAG provides substantial value by enabling access to ETH-specific knowledge
   - The gap between baseline and RAG shows the importance of retrieval-augmented generation

2. **Correctness vs Completeness**:
   - RAG improves both correctness and completeness metrics
   - Models with RAG can provide more accurate AND more complete answers
   - The improvement is consistent across both models

3. **Improvement Rate**:
   - Apertus-8B shows 45% of questions improved with RAG
   - Qwen3-8B shows 35% of questions improved with RAG
   - This indicates that smaller models may benefit more from RAG assistance

### Interpretation
The performance comparison clearly shows that RAG is effective at improving model performance on ETH-specific questions. The consistent improvement across metrics suggests that RAG successfully provides the institutional knowledge that models lack in their training data.

---

## Plot 2: Language Analysis (`language_analysis.png`)

### Description
This visualization analyzes performance by question language (English vs German) and shows how RAG affects multilingual performance.

### Key Insights

1. **Language Performance**:
   - Both English and German questions show improvement with RAG
   - RAG helps models handle multilingual queries more effectively
   - The improvement is consistent across languages

2. **Baseline Language Gaps**:
   - Baseline models show similar performance across languages
   - This suggests the models have reasonable multilingual capabilities
   - RAG enhances these capabilities by providing language-appropriate context

3. **RAG Language Benefits**:
   - RAG provides context in the same language as the question
   - This helps models generate more appropriate responses
   - Multilingual retrieval is crucial for ETH's international environment

### Interpretation
RAG successfully improves performance for both English and German questions, demonstrating that the retrieval system can provide relevant context regardless of language. This is important for ETH Zurich's multilingual environment.

---

## Plot 3: RAG Improvement Analysis (`rag_improvement_analysis.png`)

### Description
This comprehensive analysis shows winner distribution, score improvements, and correct answer counts.

### Key Insights

1. **Winner Distribution**:
   - RAG wins significantly more questions than baseline
   - Most questions show improvement or at least maintain baseline performance
   - Very few questions show degradation with RAG

2. **Score Improvement Distribution**:
   - The histogram shows a positive skew: more questions improve than degrade
   - Most improvements are moderate (0.1-0.3 score increase)
   - Some questions show dramatic improvement (>0.5 score increase)

3. **Correct Answers Count**:
   - RAG increases the number of correct/partial answers
   - Apertus-8B: ~79 correct/partial answers with RAG vs ~50 with baseline
   - Qwen3-8B: ~70 correct/partial answers with RAG vs ~50 with baseline
   - This represents a 40-60% increase in useful answers

### Interpretation
The improvement analysis demonstrates that RAG consistently improves model performance. The positive distribution of improvements and increased correct answer counts show that RAG is providing genuine value by enabling access to ETH-specific knowledge.

---

## Plot 4: Baseline Tag Distribution (`tag_distribution.png`)

### Description
This visualization shows how baseline models handle uncertainty and knowledge boundaries.

### Key Insights

1. **Tag Categories**:
   - **Correct**: Very few (0-5%) - models rarely know specific ETH tools without RAG
   - **Partial**: Some (10-20%) - models provide useful but incomplete information
   - **Generic**: Most (50-70%) - models give general advice without ETH context
   - **Refusal**: Moderate (20-35%) - models honestly admit they don't know
   - **Hallucination**: Low (5-15%) - models rarely invent false information

2. **Model Differences**:
   - Models vary in their refusal rates (honesty)
   - Models vary in their hallucination rates (safety)
   - These differences indicate different approaches to uncertainty

3. **Baseline Limitations**:
   - High generic rate shows models don't recognize need for ETH-specific knowledge
   - Low correct rate confirms models lack institutional knowledge
   - Refusal rate shows models are aware of knowledge limits

### Interpretation
The tag distribution reveals that baseline models are "institutionally blind" - they lack specific ETH knowledge. The high generic rate and low correct rate confirm that RAG is necessary to provide institutional knowledge. The variation in refusal and hallucination rates shows different safety characteristics across models.

---

## Overall Conclusions

1. **RAG is Effective**: All plots consistently show that RAG improves model performance on ETH-specific questions.

2. **Institutional Knowledge Gap**: Baseline performance is low because models lack ETH-specific knowledge, confirming the need for RAG.

3. **Consistent Improvement**: RAG improves performance across:
   - Different models (Apertus-8B, Qwen3-8B)
   - Different languages (English, German)
   - Different metrics (correctness, completeness, aggregate score)

4. **Quality Over Quantity**: While absolute scores may seem low, the improvement rate and correct answer counts demonstrate meaningful enhancement.

5. **Resource Matching Challenge**: Judge scores may be lower than expected because retrieved resources don't always perfectly match ground truth documents. This strict evaluation ensures high-quality assessment but means partial credit is common.

---

## Recommendations

1. **Continue RAG Development**: The consistent improvements justify continued investment in RAG systems.

2. **Improve Retrieval**: Better retrieval matching could further improve scores by ensuring retrieved documents more closely match ground truth.

3. **Model Selection**: Consider both baseline safety (refusal/hallucination rates) and RAG improvement when selecting models for production.

4. **Multilingual Support**: Ensure retrieval system handles both English and German queries effectively.

5. **Evaluation Refinement**: Consider adjusting evaluation criteria to better account for partial matches between retrieved documents and ground truth.


