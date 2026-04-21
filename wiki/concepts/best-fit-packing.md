---
type: concept
title: Best-Fit Packing
slug: best-fit-packing
date: 2026-04-20
updated: 2026-04-20
aliases: [best-fit data packing]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Best-Fit Packing** — a pretraining data-formatting strategy that preserves document chunks intact by assigning them to fixed-length sequences with a best-fit bin-packing heuristic.

## Key Points

- The method first segments only overlength documents into chunks of size `<= L`, avoiding unnecessary truncation for shorter documents.
- It then packs chunks into fixed-length training sequences by minimizing leftover capacity rather than concatenating raw token streams and slicing arbitrarily.
- The paper implements the packing stage with an optimized Best-Fit-Decreasing algorithm using a segment tree over remaining capacities.
- Empirically, the resulting sequences are almost as compact as concatenation, adding only `0.00063%` to `0.0028%` more sequences in the reported datasets.
- Models trained with this formatting show better context following, summarization faithfulness, and code generation accuracy than the concatenation baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
