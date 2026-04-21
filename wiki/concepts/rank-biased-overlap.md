---
type: concept
title: Rank-Biased Overlap
slug: rank-biased-overlap
date: 2026-04-20
updated: 2026-04-20
aliases: [RBO, 排名偏置重叠]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rank-Biased Overlap** (排名偏置重叠) — a top-weighted ranking similarity measure that compares two ordered result lists while putting more mass on higher ranks.

## Key Points

- The paper adds RBO to quantify how closely PLAID approximates exhaustive ColBERTv2 search, which the original PLAID work did not report.
- RBO is computed on MS MARCO Dev with persistence `p = 0.99`, making deep but still top-heavy ranking agreement visible.
- PLAID's reproduced operating points improve from `0.612` to `0.890` to `0.983` as more documents survive pruning.
- Exhaustive ColBERTv2 defines the reference ranking with `RBO = 1.000`, so lower values directly expose approximation error.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[macavaney-2024-reproducibility]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[macavaney-2024-reproducibility]].
