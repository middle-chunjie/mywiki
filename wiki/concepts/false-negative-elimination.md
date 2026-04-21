---
type: concept
title: False Negative Elimination
slug: false-negative-elimination
date: 2026-04-20
updated: 2026-04-20
aliases: [FNE, 假负例消除]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**False Negative Elimination** (假负例消除) — a contrastive-learning adjustment that removes duplicated or semantically identical samples from the negative pool so the loss does not penalize genuinely matching examples.

## Key Points

- In CVR logs, the same user-ad-context feature tuple can appear multiple times with different outcomes because of repeated exposures.
- Traditional in-batch contrastive learning would incorrectly treat those duplicates as negatives, creating contradictory supervision.
- CL4CVR defines `M(i) = {j} ∪ {k | I(o(ẽ_i), o(ẽ_k)) = 0}` so duplicate-feature examples are excluded from the denominator.
- The paper shows EM + FNE improves AUC over EM alone on both the industrial and public datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ouyang-2023-contrastive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ouyang-2023-contrastive]].
