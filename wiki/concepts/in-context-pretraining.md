---
type: concept
title: In-Context Pretraining
slug: in-context-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [context-aware pretraining, 上下文预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Context Pretraining** (上下文预训练) — a language-model pretraining strategy that orders training documents so each context contains semantically related documents, encouraging cross-document conditioning without changing the base LM objective.

## Key Points

- The paper changes only document ordering, not the autoregressive loss or the LLaMA-style model architecture.
- Related documents are found with dense retrieval and approximate nearest-neighbor search, then ordered with a graph-traversal heuristic so each document appears only once.
- The constructed contexts are meant to create useful predictive signals across document boundaries rather than packing unrelated documents together.
- The method improves in-context learning, reading comprehension, retrieval-augmented QA, factuality under knowledge conflict, and long-context reasoning relative to standard pretraining.
- Ablations show both stronger document linking and semantic deduplication matter: the final design achieves `7.3` perplexity versus `8.2` for random ordering and `8.3` without deduplication.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2024-incontext-2310-10638]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2024-incontext-2310-10638]].
