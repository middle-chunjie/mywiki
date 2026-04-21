---
type: concept
title: Distribution-Based Search
slug: distribution-based-search
date: 2026-04-20
updated: 2026-04-20
aliases: [distribution-based search, 基于分布的搜索]
tags: [search, theory]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Distribution-Based Search** (基于分布的搜索) — a search framework that evaluates algorithms by how quickly they recover a target drawn from a known or predicted probability distribution over candidates.

## Key Points

- [[fijalkow-2022-scaling]] defines the search loss as the expected index of the first correct program when the hidden target is sampled from `D`.
- The framework cleanly separates the learned prediction stage from the symbolic search stage, letting different algorithms be compared against the same induced distribution.
- It yields a simple characterization of optimal enumeration: output every program once and in non-increasing probability order.
- Within this framework, the paper introduces Heap Search for enumeration and SQRT Sampling for memoryless probabilistic search.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
