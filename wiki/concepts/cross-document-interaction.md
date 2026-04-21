---
type: concept
title: Cross-Document Interaction
slug: cross-document-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [document-document interaction, 跨文档交互]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-Document Interaction** (跨文档交互) — a ranking mechanism in which candidate documents can condition on one another within the same model context, allowing relative evidence and competition to shape each document representation.

## Key Points

- The paper's listwise prompt places multiple documents in one context window so each document can attend to the others under causal self-attention.
- This lets the reranker capture comparative context, such as supporting versus competing evidence, before final embeddings are extracted.
- Cross-document interaction is presented as the main advantage over separate-encoding late-interaction and bi-encoder methods.
- The reported ordering ablation shows modest sensitivity: random, descending, and ascending input orders reach `62.24`, `61.85`, and `61.45` average nDCG@10 on BEIR, respectively.
- The mechanism is bounded by the long-context budget, so very large candidate sets still require batching.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-jinarerankerv-2509-25085]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-jinarerankerv-2509-25085]].
