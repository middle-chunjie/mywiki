---
type: concept
title: Domain-Specific Retriever
slug: domain-specific-retriever
date: 2026-04-20
updated: 2026-04-20
aliases: [specialized retriever, 领域专用检索器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Domain-Specific Retriever** (领域专用检索器) — a retrieval model trained or adapted to one domain so that its embeddings better match the distribution and relevance structure of that domain.

## Key Points

- The paper argues that domain-specific retrievers often outperform a single general-purpose retriever on specialized test collections.
- RouterRetriever implements domain specialization through one LoRA expert per domain while keeping the same frozen base encoder.
- Scaling expert training data improves in-domain performance quickly, but does not necessarily improve out-of-domain retrieval.
- Some domains such as SciFact and NFCorpus benefit strongly from in-domain experts, while broader domains such as MSMARCO or ArguAna transfer more widely.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-routerretriever-2409-02685]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-routerretriever-2409-02685]].
