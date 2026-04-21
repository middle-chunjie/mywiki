---
type: concept
title: Depth-k Pooling
slug: depth-k-pooling
date: 2026-04-20
updated: 2026-04-20
aliases:
  - depth-k pooling
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Depth-k Pooling** — a test-set construction strategy that combines the top-ranked outputs from multiple systems up to depth `k` to create an annotation pool for relevance assessment.

## Key Points

- ProCIS uses depth-k pooling to build a more complete judged test set for proactive and reactive retrieval.
- The paper forms `5` pools of up to `10` candidates each from BM25, SPLADE, ANCE, ColBERT, and LMGR.
- Pool diversity is intentional: lexical, sparse, dense, multi-vector, and LLM-driven systems contribute candidates.
- Pooling is a key reason the final test set contains more relevant documents per conversation than the raw Reddit hyperlinks alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samarinas-2024-procis]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samarinas-2024-procis]].
