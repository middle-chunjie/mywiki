---
type: concept
title: Softmax-Off-by-One
slug: softmax-off-by-one
date: 2026-04-20
updated: 2026-04-20
aliases: [SoftMax_1, SoftMax-off-by-One]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Softmax-Off-by-One** — a softmax variant that reserves an extra unit of normalization mass, letting attention avoid assigning all probability to contextual tokens.

## Key Points

- The paper studies `SoftMax_1(x)_i = e^{x_i} / (1 + \sum_j e^{x_j})` as a possible fix for attention sinks.
- This formulation is described as equivalent to adding a token with all-zero key and value features to the attention computation.
- In the authors' `160M`-parameter experiments, the zero-sink variant improves streaming over vanilla attention but still depends on other initial tokens.
- The learnable sink token outperforms the zero-sink alternative, indicating that a trained repository for excess attention is more effective than a fixed normalization trick.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
