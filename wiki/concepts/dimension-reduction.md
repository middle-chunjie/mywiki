---
type: concept
title: Dimension Reduction
slug: dimension-reduction
date: 2026-04-20
updated: 2026-04-20
aliases: [dimensionality reduction, vector compression, 降维]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dimension Reduction** (降维) — transforming high-dimensional representations into a lower-dimensional space to reduce storage and computation while retaining useful structure.

## Key Points

- The paper applies dimension reduction to datastore keys because `1024`- or `1536`-dimensional vectors dominate memory and distance-computation cost.
- Smaller vector dimensions speed up both ANN search and datastore storage.
- Aggressive compression below `128` dimensions hurts perplexity badly, but moderate compression at `256` or `512` dimensions remains competitive.
- In this study, dimension reduction is one of the strongest efficiency levers and can even improve perplexity relative to the original vectors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2021-efficient-2109-04212]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2021-efficient-2109-04212]].
