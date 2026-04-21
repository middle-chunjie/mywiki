---
type: concept
title: Exact Substring Deduplication
slug: exact-substring-deduplication
date: 2026-04-20
updated: 2026-04-20
aliases: [exact substring dedup, exact span deduplication, 精确子串去重]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exact Substring Deduplication** (精确子串去重) — a deduplication procedure that identifies character- or token-exact repeated spans across documents and removes or masks those spans directly.

## Key Points

- [[penedo-2023-refinedweb-2306-01116]] applies exact substring deduplication after MinHash so that fuzzy near-duplicates are removed before the more memory-intensive exact stage.
- The method uses suffix arrays to find repeated spans efficiently in a concatenated corpus representation.
- Only duplicated spans of at least `50` tokens are targeted, and the chosen production setting is `EXACTSUBSTR-CUT`.
- Documents with fewer than `20` remaining non-duplicated characters are discarded after cutting duplicated spans.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[penedo-2023-refinedweb-2306-01116]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[penedo-2023-refinedweb-2306-01116]].
