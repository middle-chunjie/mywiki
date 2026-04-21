---
type: concept
title: Token Edit Distance
slug: token-edit-distance
date: 2026-04-20
updated: 2026-04-20
aliases: [TED, token-level edit distance, token 编辑距离]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Token Edit Distance** (token 编辑距离) — an edit-distance metric over token sequences that counts the additions, deletions, and substitutions needed to transform a model output into the target sequence.

## Key Points

- [[schaeffer-2023-emergent-2304-15004]] uses token edit distance as a smoother alternative to exact accuracy for analyzing scaling behavior.
- Under the paper's approximation, expected token edit distance scales roughly as `L * (1 - p(correct token))`, making it approximately linear in per-token error.
- Because it is not thresholded on all tokens being correct, it reveals graded improvements that exact-match metrics can hide.
- On GPT-3/InstructGPT arithmetic tasks, switching from accuracy to token edit distance makes allegedly emergent curves become smooth and predictable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[schaeffer-2023-emergent-2304-15004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[schaeffer-2023-emergent-2304-15004]].
