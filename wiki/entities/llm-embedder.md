---
type: entity
title: LLM-Embedder
slug: llm-embedder
date: 2026-04-20
entity_type: tool
aliases: [LLM Embedder, FlagEmbedding LLM-Embedder]
tags: [embedding, retrieval, multi-task]
---

## Description

LLM-Embedder is a unified dense text embedding model developed by [[baai]] and [[renmin-university-of-china]], released under [[flagembedding]]. It is initialized from BGE-base and fine-tuned to support four LLM retrieval augmentation modes: knowledge enhancement, long-context memory retrieval, in-context example retrieval, and tool retrieval — in a single model.

## Key Contributions

- First unified embedding model to comprehensively cover all major LLM retrieval augmentation needs in one checkpoint.
- Introduces stabilized distillation and homogeneous in-batch negative sampling as key training innovations for multi-task embedding.
- Outperforms both general-purpose and task-specific retrievers across all five evaluated scenarios.

## Related Concepts

- [[unified-embedding-model]]
- [[retrieval-augmented-generation]]
- [[text-embedding]]
- [[knowledge-distillation]]
- [[in-batch-negative-sampling]]

## Sources

- [[zhang-2023-retrieve-2310-07554]]
