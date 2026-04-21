---
type: concept
title: Layer-Selective Rank Reduction
slug: layer-selective-rank-reduction
date: 2026-04-20
updated: 2026-04-20
aliases: [LASER]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Layer-Selective Rank Reduction** — a post-training intervention that replaces a chosen model weight matrix at a chosen layer with a truncated low-rank approximation in order to alter behavior without retraining.

## Key Points

- The paper parameterizes LASER by `(τ, ℓ, ρ)`, selecting matrix type, layer, and retained-rank fraction.
- The strongest gains come from reducing later-layer MLP matrices, especially `U_in`, rather than applying uniform compression everywhere.
- LASER improves factual QA, paraphrase robustness, and several reasoning benchmarks without adding parameters or using extra data.
- The intervention can be composed across multiple layers through a greedy search over promising `(τ, ℓ, ρ)` settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sharma-2023-truth-2312-13558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sharma-2023-truth-2312-13558]].
