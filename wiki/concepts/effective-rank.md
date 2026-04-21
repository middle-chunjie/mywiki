---
type: concept
title: Effective Rank
slug: effective-rank
date: 2026-04-20
updated: 2026-04-20
aliases: [effective dimensionality, 有效秩]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Effective Rank** (有效秩) — the exponential of the entropy of a matrix's normalized singular or eigenvalue spectrum, used as a soft measure of dimensionality or diversity.

## Key Points

- The paper identifies Vendi Score as the effective rank of the kernel similarity matrix.
- This view explains why the metric behaves like an "effective number" rather than a hard count of distinct modes.
- Effective-rank computation also motivates the lower-cost implementation path when explicit feature embeddings are available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
