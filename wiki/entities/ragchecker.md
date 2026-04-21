---
type: entity
title: RagChecker
slug: ragchecker
date: 2026-04-20
entity_type: tool
aliases: [RAGChecker]
tags: [framework]
---

## Description

RagChecker is the fine-grained evaluation framework introduced by this paper for diagnosing retrieval and generation behavior in RAG systems. It computes overall, retriever, and generator metrics from claim-level entailment checks.

## Key Contributions

- Defines claim-level precision, recall, and F1 for end-to-end RAG quality.
- Adds retriever diagnostics such as claim recall and context precision.
- Adds generator diagnostics such as context utilization, noise sensitivity, hallucination, self-knowledge, and faithfulness.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[claim-verification]]
- [[context-utilization]]

## Sources

- [[ru-2024-ragchecker-2408-08067]]
