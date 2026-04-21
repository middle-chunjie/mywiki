---
type: concept
title: Approximate Nearest-Neighbor Search
slug: approximate-nearest-neighbor-search
date: 2026-04-20
updated: 2026-04-20
aliases: [ANN search, ANN retrieval, 近似最近邻搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Approximate Nearest-Neighbor Search** (近似最近邻搜索) — a similarity-search procedure that sacrifices exactness to retrieve near neighbors much faster on large vector collections.

## Key Points

- The paper treats ANN search as the practical retrieval mechanism underlying `kNN-LM` because exact search over a `103M`-entry datastore is too slow.
- FAISS is used as a black-box ANN backend that combines indexing and vector quantization.
- ANN returns approximate distances, and the paper uses these directly rather than recomputing exact full-precision distances from disk.
- The work aims to improve efficiency in an index-agnostic way, complementing rather than replacing ANN system design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2021-efficient-2109-04212]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2021-efficient-2109-04212]].
