---
type: concept
title: Locality-Sensitive Hashing
slug: locality-sensitive-hashing
date: 2026-04-20
updated: 2026-04-20
aliases: [LSH, 局部敏感哈希]
tags: [hashing, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Locality-Sensitive Hashing** (局部敏感哈希) — a hashing strategy designed so that similar inputs have a higher probability of being mapped to the same discrete code.

## Key Points

- The paper includes LSHI as a retrieval-indexing baseline in which each ID digit is derived from Boolean LSH codes.
- Its implementation maps every fifth hash bit into `V = [1,\ldots,32]` and appends extra HKmI digits to resolve collisions.
- Unlike hierarchical `k`-means, LSHI does not provide semantically meaningful prefixes for identifiers.
- LSHI consistently underperforms BMI on both NQ320K and MARCO Lite in Rec@1 and MRR.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
