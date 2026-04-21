---
type: entity
title: ELSER
slug: elser
date: 2026-04-20
entity_type: tool
aliases: [Elastic Learned Sparse Encoder, ELSERv1]
tags: []
---

## Description

ELSER is Elastic's learned sparse retrieval model used in [[katsis-2025-mtrag-2501-03468]] for corpus indexing, annotation-time retrieval, and the paper's main full-RAG retrieval pipeline.

## Key Contributions

- Serves as the sparse retriever used to build and evaluate mtRAG.
- Achieves the best retrieval results in the paper, especially when combined with [[query-rewriting]].

## Related Concepts

- [[sparse-retrieval]]
- [[query-rewriting]]
- [[information-retrieval]]

## Sources

- [[katsis-2025-mtrag-2501-03468]]
