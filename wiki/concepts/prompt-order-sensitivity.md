---
type: concept
title: Prompt Order Sensitivity
slug: prompt-order-sensitivity
date: 2026-04-20
updated: 2026-04-20
aliases: [demonstration order sensitivity, 提示顺序敏感性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Order Sensitivity** (提示顺序敏感性) — the phenomenon that reordering the same in-context demonstrations can substantially change a language model's predictions and accuracy.

## Key Points

- The paper enumerates permutations of fixed demonstrations and shows large accuracy variance even when the demonstration set is unchanged.
- Order sensitivity persists after post-calibration, so calibration alone does not remove prompt instability.
- T-fair-Prompting addresses order only indirectly by placing fairer demonstrations later in the context.
- G-fair-Prompting explicitly accounts for order by greedily inserting the next demonstration to maximize prompt fairness.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-fairnessguided-2303-13217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-fairnessguided-2303-13217]].
