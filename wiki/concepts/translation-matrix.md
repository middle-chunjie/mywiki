---
type: concept
title: Translation Matrix
slug: translation-matrix
date: 2026-04-20
updated: 2026-04-20
aliases: [翻译矩阵, matching matrix]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Translation Matrix** (翻译矩阵) — a matrix of pairwise query-document term similarities used to represent potential exact and soft matches between the two texts.

## Key Points

- In K-NRM each entry is `M_{ij} = cos(v_{t_i^q}, v_{t_j^d})`, computed from learned token embeddings.
- The matrix serves as the bridge between token embeddings and ranking features, replacing hand-crafted retrieval signals.
- Kernel pooling operates row-wise over this matrix to convert raw similarities into soft-match counts for each query term.
- The paper frames this as a neuralized version of earlier statistical translation ideas for information retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2017-endtoend-1706-06613]].
