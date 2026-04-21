---
type: concept
title: Maximum Mean Discrepancy
slug: maximum-mean-discrepancy
date: 2026-04-20
updated: 2026-04-20
aliases: [MMD, 最大均值差异]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Maximum Mean Discrepancy** (最大均值差异) — a kernel-based two-sample distance that measures how different two probability distributions are after embedding them in a reproducing-kernel Hilbert space.

## Key Points

- Lumina uses squared MMD to compare next-token distributions conditioned on relevant versus random documents.
- The paper rewrites MMD directly over token-embedding distributions induced by vocabulary probabilities.
- Its default kernel is cosine, although RBF variants with different bandwidths are also tested.
- For efficiency, the implementation approximates the MMD sum with the top `100` vocabulary tokens.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
