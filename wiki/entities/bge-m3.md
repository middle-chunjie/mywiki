---
type: entity
title: BGE-M3
slug: bge-m3
date: 2026-04-20
entity_type: tool
aliases: [BGE M3, BGE-M3-Embedding]
tags: []
---

## Description

BGE-M3 is the dense retrieval model used in [[qian-2024-memorag-2409-05591]] as the default retriever for MemoRAG, HyDE, and RQ-RAG, and it is also evaluated as a standard RAG baseline.

## Key Contributions

- Provides the top-`3` dense retrieval step over `512`-token context chunks in MemoRAG's main experiments.
- Serves as a strong retrieval baseline that MemoRAG substantially outperforms on the paper's aggregate benchmark score.

## Related Concepts

- [[dense-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[retrieval-augmented-generation]]

## Sources

- [[qian-2024-memorag-2409-05591]]
