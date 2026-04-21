---
type: concept
title: Bin Packing
slug: bin-packing
date: 2026-04-20
updated: 2026-04-20
aliases: [bin packing, 装箱问题]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bin Packing** (装箱问题) — a combinatorial optimization problem that assigns items of varying sizes to fixed-capacity bins while minimizing the number of bins used.

## Key Points

- The paper maps document chunks to items and training sequences of length `L` to bins.
- Its objective is to find a partition `S` that minimizes the number of training sequences `M` subject to each sequence holding at most `L` tokens.
- Best-Fit-Decreasing is used as the practical approximation because exact optimization is NP-hard.
- The authors exploit the fact that chunk lengths are integers in `[1, L]` to reduce the search structure from `O(N)` bins to `O(L)` capacities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
