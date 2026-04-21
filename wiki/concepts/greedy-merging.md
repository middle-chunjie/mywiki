---
type: concept
title: Greedy Merging
slug: greedy-merging
date: 2026-04-20
updated: 2026-04-20
aliases: [local merge compression, 贪心合并]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Greedy Merging** (贪心合并) — a local datastore-compression strategy that merges nearby entries sharing the same target token without performing full global clustering.

## Key Points

- The algorithm iterates through datastore entries and inspects a small number of nearest neighbors `K` to decide whether two entries should be merged.
- A merge happens only when a neighbor has the same target token and has not already been merged away, which limits approximation error.
- Each surviving entry stores a weight `s_i` indicating how many original records it represents.
- Among the pruning methods in the paper, greedy merging gives the best perplexity-speed tradeoff on WikiText-103.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2021-efficient-2109-04212]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2021-efficient-2109-04212]].
