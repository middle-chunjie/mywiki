---
type: concept
title: Structured Pruning
slug: structured-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [structured model pruning, 结构化剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structured Pruning** (结构化剪枝) — a model compression method that removes whole architectural substructures such as layers, heads, or channels rather than isolated individual weights.

## Key Points

- The paper uses structured pruning not just for deployment compression, but as the first stage of producing a new smaller general-purpose LLM.
- Pruning decisions are applied at multiple granularities: layers, hidden dimensions, attention heads, and FFN intermediate dimensions.
- The authors argue that for LLMs, pruning alone causes substantial capability loss, so recovery via continued pre-training is necessary.
- Compared with non-uniform structured pruning baselines such as CoFiPruning, the work emphasizes dense architectures that remain efficient at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2024-sheared-2310-06694]].
