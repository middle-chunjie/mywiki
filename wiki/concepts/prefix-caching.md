---
type: concept
title: Prefix Caching
slug: prefix-caching
date: 2026-04-20
updated: 2026-04-20
aliases: [prefix caching, 前缀缓存]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prefix Caching** (前缀缓存) — reusing the encoded representation of a shared prompt prefix so repeated long-context queries do not require re-encoding the entire prefix each time.

## Key Points

- CiC prompting places the corpus before the query so the corpus can function as a reusable prefix.
- This design makes long-context prompting more analogous to indexing in retrieval systems, where the corpus is processed once and reused across queries.
- The paper argues that putting the query at the end is both cheaper and more accurate than conditioning the corpus on each query from scratch.
- The authors could not directly measure caching gains through the public APIs they used, so the efficiency benefit remains an unverified systems expectation in this source.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-can-2406-13121]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-can-2406-13121]].
