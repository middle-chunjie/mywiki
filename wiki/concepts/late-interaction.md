---
type: concept
title: Late Interaction
slug: late-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [late-interaction retrieval, 延迟交互]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Late Interaction** (延迟交互) — a retrieval design in which query and document tokens are encoded separately and only combined at scoring time through token-level similarity aggregation.

## Key Points

- ColBERT is the paper's main late-interaction baseline, using token-level max aggregation rather than a single-vector dot product.
- Late interaction improves retrieval expressivity because it keeps contextualized token evidence available until ranking.
- The downside emphasized here is serving complexity: exact late interaction normally requires loading many candidate-document token vectors.
- XTR keeps the late-interaction modeling benefit but approximates the expensive scoring stage using retrieved token scores and imputation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-nd-rethinking]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-nd-rethinking]].
