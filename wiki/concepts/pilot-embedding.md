---
type: concept
title: Pilot Embedding
slug: pilot-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [pilot embeddings, 引导嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pilot Embedding** (引导嵌入) — a representative centroid embedding used to estimate which expert retriever is most appropriate for a query before full retrieval encoding.

## Key Points

- RouterRetriever constructs pilot embeddings by first finding the best-performing expert for each training instance and then centroiding base-encoder embeddings within each expert-assigned group.
- Each domain dataset can contribute up to one pilot embedding per expert, so the pilot library has at most `T^2` embeddings for `T` experts and `T` training datasets.
- At inference time, query routing depends on similarities between the query's base embedding and these pilot embeddings.
- The appendix reports that using a single centroid per group (`k = 1`) works better than multiple centroids, because additional centroids become distractors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-routerretriever-2409-02685]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-routerretriever-2409-02685]].
