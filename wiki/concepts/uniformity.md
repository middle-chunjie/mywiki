---
type: concept
title: Uniformity
slug: uniformity
date: 2026-04-20
updated: 2026-04-20
aliases: [均匀性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Uniformity** (均匀性) — the property that embedding vectors are broadly and evenly distributed over the representation space rather than collapsing into a narrow region.

## Key Points

- SimCSE adopts the metric `` `l_uniform = log E exp(-2 ||f(x) - f(y)||^2)` `` from Wang and Isola to analyze sentence embedding quality.
- Unsupervised SimCSE improves uniformity substantially relative to vanilla pre-trained encoders while preserving acceptable alignment.
- The paper argues that much of contrastive learning's benefit comes from the negative-pair term, which encourages a flatter singular-value spectrum and thus better uniformity.
- Post-processing baselines can improve uniformity too, but SimCSE claims better overall results because it jointly improves uniformity and positive-pair alignment.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2022-simcse-2104-08821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2022-simcse-2104-08821]].
