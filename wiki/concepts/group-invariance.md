---
type: concept
title: Group Invariance
slug: group-invariance
date: 2026-04-20
updated: 2026-04-20
aliases: [group invariance, 群不变性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Group Invariance** (群不变性) — the property that a function's output remains unchanged when any transformation from a specified group acts on its input.

## Key Points

- The paper defines the end goal of code analysis as `p(r(g(c))) = p(r(c))` for all semantics-preserving transformations `g`.
- In SymC, invariance is implemented at the prediction stage rather than enforced on the early representation itself.
- Mean pooling and token-level prediction heads are used because they are invariant to the relevant automorphism-induced permutations.
- The framework ties invariant outputs to semantic robustness on tasks such as function similarity, signature prediction, and memory-region prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
