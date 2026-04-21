---
type: concept
title: Intrinsic Dimension
slug: intrinsic-dimension
date: 2026-04-20
updated: 2026-04-20
aliases: [intrinsic dimensionality, effective degrees of freedom]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Intrinsic Dimension** (内在维度) — the effective number of degrees of freedom needed to describe the local geometry of an embedding space.

## Key Points

- The paper uses intrinsic dimension to diagnose whether a corpus embedding space is well-behaved for dense retrieval.
- It estimates this quantity with the TwoNN method from nearest-neighbor distance ratios `` `\mu = r_2 / r_1` ``.
- High intrinsic dimension is interpreted as worsening the curse of dimensionality and weakening cosine-based retrieval signals.
- The reported values differ across corpora, providing one of the offline signals that a router can use when choosing between vector, graph, and hybrid retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
