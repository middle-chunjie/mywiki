---
type: concept
title: Multi-Granularity Retrieval
slug: multi-granularity-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-level retrieval, 多粒度检索]
tags: [retrieval, chunking, rag]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Granularity Retrieval** (多粒度检索) — a retrieval paradigm in which documents are indexed at multiple granularity levels simultaneously (e.g., sentence, paragraph, section), and queries are matched against all levels to identify the most informative unit size for a given information need.

## Key Points

- A corpus indexed at `n_gra` levels stores overlapping but non-redundant chunk hierarchies: each level-j chunk contains exactly 2 adjacent level-(j-1) chunks (non-overlapping doubling scheme in MoG).
- Coarser granularities tend to yield higher raw similarity scores due to larger text coverage, making direct score comparison across levels biased; this necessitates a normalized or routing-based selection strategy.
- Retrieving `k_r` candidates per granularity produces a pool of `n_gra × k_r` candidates, traded against precision via a router or reranker.
- In the [[mix-of-granularity]] framework, the finest-grained chunks serve as the common anchor (they appear at every level), enabling cross-level comparison without direct score normalization.
- Storage cost scales linearly with `n_gra` (each level is a separate index), approximately 2.7× for 5 levels including embeddings.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2024-mixofgranularity-2406-00456]].
