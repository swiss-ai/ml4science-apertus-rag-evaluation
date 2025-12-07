# Apertus RAG Architecture Evaluation

Project explores the design and evaluation of modern Retrieval-Augmented Generation (RAG) architectures for Apertus, focusing on how different retrieval, chunking, and reranking strategies affect system performance and answer quality. It aims to systematically test combinations of search (ElasticSearch, BM25, vector search), synthetic query generation, embedder model, rerankers, and metadata injection.

## Key Objectives

- **Comparing retrieval effectiveness** across search strategies
- **Measuring performance impact** of query expansion and synthetic query generation
- **Evaluating chunking strategies** and metadata enrichment for context precision
- **Testing reranking** as a lightweight method for boosting retrieval accuracy
- **Assessing trade-offs** between accuracy, latency, and cost for production-scale RAG

## Technologies

- LlamaIndex
- Elasticsearch
- Mistral AI
- BM25
- Vector Search
