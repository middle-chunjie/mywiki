---
type: concept
title: Power-Law Scaling
slug: power-law-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [Scaling Law, 幂律缩放]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Power-Law Scaling** (幂律缩放) — an empirical regularity in which model performance changes predictably as a power function of resource variables such as parameter count or dataset size.

## Key Points

- The paper fits dense-retrieval model quality with `` `L(N) = (A / N)^alpha + delta_N` ``, where `N` is the number of non-embedding parameters.
- It also fits annotation scaling as `` `L(D) = (B / D)^beta + delta_D` ``, where `D` is the number of annotated query-passage pairs.
- Reported fits are strong on both MS MARCO and T2Ranking, with `R^2` ranging from `0.971` to `0.999`.
- The non-zero asymptotes `` `delta_N` `` and `` `delta_D` `` are motivated by incomplete binary relevance labels and false negatives in retrieval benchmarks.
- A joint fit over model size and data size is later combined with explicit annotation, training, and inference cost terms for budget planning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2024-scaling-2403-18684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2024-scaling-2403-18684]].
