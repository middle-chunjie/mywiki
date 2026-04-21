---
type: concept
title: Compositional Control
slug: compositional-control
date: 2026-04-20
updated: 2026-04-20
aliases: [compositional control, 组合控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compositional Control** (组合控制) — the ability to combine multiple learned control signals so generation reflects more than one attribute at the same time.

## Key Points

- LM-Switch composes multiple controls by adding their switch matrices, decoding with `M(ε_1W_1 + ε_2W_2)`.
- Theorem 3 gives a bounded approximation error for combining two switch matrices in this additive way.
- The paper demonstrates joint control of sentiment and toxicity, with sentiment swept over `[-5ε_0, 5ε_0]` and toxicity over `[0, 5ε_0]`.
- The results show compositionality is useful but not perfectly factorized, since sentiment and toxicity can still influence one another.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
