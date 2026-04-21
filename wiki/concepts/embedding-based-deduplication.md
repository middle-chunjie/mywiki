---
type: concept
title: Embedding-Based Deduplication
slug: embedding-based-deduplication
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic deduplication, 基于嵌入的去重]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Embedding-Based Deduplication** (基于嵌入的去重) — a semantic deduplication procedure that removes items whose vector representations are too similar under a metric such as cosine similarity.

## Key Points

- The paper uses embedding-based deduplication after surface-form deduplication with MinHash.
- Persona descriptions are embedded with a text embedding model such as `text-embedding-3-small`.
- Items with cosine similarity greater than `0.9` are filtered out in the reported setup.
- The threshold is treated as tunable: lower thresholds can enforce higher diversity when the desired dataset size is smaller.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chan-2024-scaling-2406-20094]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chan-2024-scaling-2406-20094]].
