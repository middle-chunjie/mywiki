---
type: concept
title: Context Precision
slug: context-precision
date: 2026-04-20
updated: 2026-04-20
aliases: [上下文精度]
tags: [rag, retrieval, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Precision** (上下文精度) — the fraction of retrieved chunks that contain at least one ground-truth claim needed by the answer.

## Key Points

- RAGChecker labels a chunk as relevant if it entails any ground-truth claim, then defines context precision as ``|{r-chunk_j}| / k`` over the top-`k` retrieved chunks.
- The paper uses a chunk-level precision rather than claim-level precision because chunked retrieval is the actual control surface in practical RAG systems.
- On the main benchmark, E5-Mistral improves average context precision from `52.3` to `61.8` relative to BM25.
- Higher overlap can slightly increase context precision by retrieving more similarly useful chunks, even when total claim coverage barely changes.
- Under a fixed context budget, larger chunks can improve context precision because fewer retrieved units are purely irrelevant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ru-2024-ragchecker-2408-08067]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ru-2024-ragchecker-2408-08067]].
