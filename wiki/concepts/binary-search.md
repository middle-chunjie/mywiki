---
type: concept
title: Binary Search
slug: binary-search
date: 2026-04-20
updated: 2026-04-20
aliases: [binary search, 二分搜索]
tags: [algorithm, search]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Binary Search** (二分搜索) — a divide-and-conquer algorithm that repeatedly halves a search interval to localize a target condition efficiently.

## Key Points

- The paper applies binary search to a sampled reasoning rollout in order to find the first incorrect step instead of evaluating every step sequentially.
- At each midpoint, the method checks whether the partial solution still admits any correct completion under Monte Carlo rollouts.
- This reduces the search complexity for error localization from `` `O(kM)` `` to `` `O(k log M)` `` when a solution has `M` steps and each probe uses `k` rollouts.
- Prefixes encountered before the first detected error become reusable states in OmegaPRM's tree and later serve as PRM supervision examples.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[luo-2024-improve-2406-06592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[luo-2024-improve-2406-06592]].
