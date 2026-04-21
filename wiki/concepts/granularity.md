---
type: concept
title: Granularity
slug: granularity
date: 2026-04-20
updated: 2026-04-20
aliases: [expert granularity, 粒度]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Granularity** (粒度) — in this fine-grained MoE setting, the factor `G = d_ff / d_expert` that controls how small each expert is relative to the dense feed-forward layer.

## Key Points

- The paper defines granularity as the ratio between the dense FFN hidden width and the expert hidden width.
- Increasing `G` makes each expert smaller while routing each token to `G` experts, keeping active parameters approximately constant.
- For fixed model size and token budget, higher granularity usually lowers loss according to an approximate power law.
- Compute-optimal `G` increases with training budget, reaching `32` or `64` for the largest projected regimes.
- Very large `G` can become inefficient in practice because routing overhead grows with the number of granular experts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[krajewski-2024-scaling-2402-07871]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[krajewski-2024-scaling-2402-07871]].
