---
type: concept
title: Round-Trip Consistency
slug: round-trip-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval consistency filtering, 往返一致性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Round-Trip Consistency** (往返一致性) — a filtering criterion that keeps a synthetic query only if a retrieval model can return the original source document for that query.

## Key Points

- PROMPTAGATOR trains an initial retriever on noisy synthetic pairs and reuses that retriever as the filter instead of depending on an external QA model.
- A generated pair `(q, d)` is kept only when `d` appears in the top `K = 1` retrieved results for `q`.
- The filtering step mainly removes generic queries, ambiguous queries, and hallucinated queries that are not well grounded in the source passage.
- The paper reports that this consistency filter improves average BEIR nDCG@10 by about `2.5` points and helps on `8/11` datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dai-2022-promptagator-2209-11755]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dai-2022-promptagator-2209-11755]].
