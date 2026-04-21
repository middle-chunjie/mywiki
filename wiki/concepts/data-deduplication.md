---
type: concept
title: Data Deduplication
slug: data-deduplication
date: 2026-04-20
updated: 2026-04-20
aliases: [dataset deduplication, 数据去重]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Deduplication** (数据去重) — the removal of exact or near-duplicate training examples to reduce repeated content and potentially lower memorization.

## Key Points

- Pythia trains a matched second suite on a near-deduplicated version of the Pile using MinHashLSH with threshold `0.87`.
- The deduplicated corpus shrinks to `≈ 207B` tokens from the original `≈ 300B` target training budget.
- Because the deduplicated corpus is smaller, the deduplicated models are trained for `≈ 1.5` epochs over that data.
- The paper reports no clear language-modeling benefit from deduplication on its evaluation suite, despite common assumptions that deduplication should help.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
