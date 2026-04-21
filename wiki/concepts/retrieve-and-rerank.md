---
type: concept
title: Retrieve-and-Rerank
slug: retrieve-and-rerank
date: 2026-04-20
updated: 2026-04-20
aliases: [检索后重排, reranking pipeline]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Retrieve-and-Rerank** (检索后重排) — a two-stage retrieval pipeline that first gathers a candidate set with a cheap retriever and then reorders those candidates using a more expensive relevance model.

## Key Points

- The paper treats dual-encoder or tf-idf retrieval followed by cross-encoder reranking as the main baseline family `RnR_X`.
- Its central critique is that the initial retriever is decoupled from the cross-encoder, which can lower recall by omitting strong items before reranking.
- Axn is proposed as a stronger alternative because it uses exact cross-encoder feedback to iteratively refine the query representation during retrieval.
- The reported improvements over `RnR_DEsrc` on YuGiOh show that better candidate generation alone can materially improve final `k`-NN recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
