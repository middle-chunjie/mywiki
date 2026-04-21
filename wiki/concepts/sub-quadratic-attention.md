---
type: concept
title: Sub-Quadratic Attention
slug: sub-quadratic-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [subquadratic attention, 次二次注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sub-Quadratic Attention** (次二次注意力) — an attention design whose computational growth with sequence length is asymptotically better than the `O(T^2)` cost of dense full-sequence self-attention.

## Key Points

- MEGABYTE derives an attention cost of `O(T^2 / P^2 + T·P)` by splitting a length-`T` byte sequence into patches of size `P`.
- Choosing `P = T^{1/3}` yields `O(T^{4/3})`, which the paper uses as its headline complexity result for long-sequence decoding.
- The savings are not only from attention sparsification: the patch-based formulation also shifts more feed-forward compute to coarse patch positions.
- The method is demonstrated on contexts ranging from `8,192` bytes in text experiments to more than `1.2M` bytes on ImageNet640.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
