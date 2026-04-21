---
type: concept
title: Suffix Array
slug: suffix-array
date: 2026-04-20
updated: 2026-04-20
aliases: [后缀数组]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Suffix Array** (后缀数组) — an array of starting positions that sorts all suffixes of a token sequence lexicographically, enabling efficient substring counting and lookup.

## Key Points

- Infini-gram uses suffix arrays instead of explicit `n`-gram count tables so arbitrary-length `n`-gram queries remain feasible at trillion-token scale.
- The implementation stores tokenized corpus bytes plus suffix-array pointers for a combined storage cost of about `7N` bytes.
- `n`-gram counting reduces to finding the first and last positions of a query string within one contiguous suffix-array segment via binary search.
- The method shards the corpus and parallelizes shard-level search so suffix-array lookup stays practical even when indexes do not fit in RAM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-infinigram-2401-17377]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-infinigram-2401-17377]].
