---
type: concept
title: Soft Matching
slug: soft-matching
date: 2026-04-20
updated: 2026-04-20
aliases: [软匹配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Matching** (软匹配) — a learned matching mechanism that retrieves semantically similar patterns without requiring exact lexical overlap.

## Key Points

- [[fang-2021-guided]] uses a neural soft-matching module to locate clue words in unseen sentences even when the wording differs from the collected clue inventory.
- The module is intended to generalize beyond regular-expression matching, which fails on paraphrases and lexical variants.
- The detected clue-word span is used to derive relative position features for the position-aware attention module.
- Reported hyperparameters include matching threshold `θ = 0.75`, maximum window size `3`, and training until loss plateaus for about `20` epochs.
- The ablation against regex matching shows that neural soft matching improves recall while keeping precision competitive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]
- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
