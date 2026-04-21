---
type: concept
title: Joint Optimization
slug: joint-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [JO, 联合优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Joint Optimization** (联合优化) — an optimization strategy that updates multiple coupled variable blocks together within a single objective rather than alternating across them.

## Key Points

- In this paper, JO updates the site-selection variables `z` and site-perturbation variables `u` together after continuous relaxation.
- The method uses PGD steps followed by projection back into the feasible set defined by the attack budget and per-site constraints.
- JO is computationally cheaper per iteration than AO because it covers both variable blocks in one pass.
- Empirically, JO is weaker than AO on the reported benchmarks and often needs about `10` iterations to approach a local optimum.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
