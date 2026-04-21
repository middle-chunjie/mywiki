---
type: concept
title: MinHash
slug: minhash
date: 2026-04-20
updated: 2026-04-20
aliases: [minimum hash, 最小哈希]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**MinHash** (最小哈希) — a locality-sensitive hashing technique that estimates Jaccard similarity between discrete feature sets through compact randomized signatures.

## Key Points

- The paper uses MinHash as the first deduplication pass over persona descriptions.
- Because persona descriptions are short, the implementation uses `1`-gram features rather than longer n-grams.
- The MinHash signature size is set to `128`.
- Personas with estimated similarity above `0.9` are removed before embedding-based semantic deduplication.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chan-2024-scaling-2406-20094]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chan-2024-scaling-2406-20094]].
