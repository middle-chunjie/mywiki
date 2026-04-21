---
type: concept
title: Information Flow
slug: information-flow
date: 2026-04-20
updated: 2026-04-20
aliases: [信息流]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Flow** (信息流) — the directed movement of task-relevant signal between token representations across layers of a neural model.

## Key Points

- This paper operationalizes information flow with attention saliency scores derived from `A ⊙ ∂L/∂A`.
- It separates text-to-label flow `S_wp`, label-to-target flow `S_pq`, and residual word-to-word flow `S_ww`.
- The reported pattern is stage-dependent: shallow layers emphasize aggregation into label words, while deep layers emphasize extraction from them.
- The same information-flow view motivates both causal intervention experiments and practical methods such as anchor re-weighting and Hiddenanchor compression.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-label]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-label]].
