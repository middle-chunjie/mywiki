---
type: concept
title: Adaptive Computation
slug: adaptive-computation
date: 2026-04-20
updated: 2026-04-20
aliases: [自适应计算]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Computation** (自适应计算) — a family of inference methods that vary compute allocation across inputs or decoding steps according to estimated difficulty.

## Key Points

- The paper positions adaptive-computation baselines as the closest prior family for speeding up autoregressive inference.
- Earlier methods often use smaller models, early exits, or input subset selection only on easier steps.
- Leviathan et al. argue that many adaptive methods require architecture or training changes and usually do not preserve identical outputs.
- Speculative decoding is presented as complementary: it also exploits easy steps, but through parallel verification rather than heuristic early stopping.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[leviathan-2023-fast-2211-17192]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[leviathan-2023-fast-2211-17192]].
