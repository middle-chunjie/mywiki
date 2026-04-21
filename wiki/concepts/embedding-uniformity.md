---
type: concept
title: Embedding Uniformity
slug: embedding-uniformity
date: 2026-04-20
updated: 2026-04-20
aliases: [嵌入均匀性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Embedding Uniformity** (嵌入均匀性) — the extent to which learned representations spread evenly across the embedding hypersphere rather than collapsing into crowded regions.

## Key Points

- The paper uses the uniformity metric `` `log E[e^{-t ||f(x)-f(y)||_2^2}]` `` to quantify how evenly embeddings are distributed.
- Lower temperatures in ordinary contrastive loss produce more uniform embeddings by imposing stronger penalties on nearby negatives.
- Greater uniformity improves separability, but the paper argues that excessive uniformity can destroy local semantic structure.
- Explicit hard negative sampling keeps uniformity relatively high even when the temperature is increased.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2021-understanding-2012-09740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2021-understanding-2012-09740]].
