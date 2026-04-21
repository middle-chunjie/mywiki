---
type: concept
title: Dual-Encoder Retrieval
slug: dual-encoder-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [bi-encoder retrieval, 双编码器检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dual-Encoder Retrieval** (双编码器检索) — a retrieval architecture that encodes queries and passages independently into the same embedding space and scores them with a simple similarity function such as a dot product.

## Key Points

- HAConvDR is implemented on top of ANCE, a dual-encoder dense retriever, and scores query-passage pairs with `` `S(q, p) = F_Q(q)^T · F_P(p)` ``.
- The paper updates only the query encoder during conversational training while freezing the passage encoder.
- Additional historical pseudo positives and historical hard negatives are injected into the standard dual-encoder contrastive objective.
- Faiss is used as the dense retrieval backend for approximate nearest-neighbor search during experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-historyaware-2401-16659]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-historyaware-2401-16659]].
