---
type: concept
title: Speculative Retrieval
slug: speculative-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [投机检索, cache-based speculative retrieval]
tags: [retrieval, serving, speculation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Speculative Retrieval** (投机检索) — a retrieval acceleration technique that substitutes expensive knowledge-base queries with fast local-cache lookups during iterative generation, deferring ground-truth verification to a subsequent batched retrieval step.

## Key Points

- Maintains a per-request local cache seeded by the initial knowledge-base retrieval; subsequent retrievals query the cache using the same scoring metric (e.g., inner product for dense retrievers, BM25 score for sparse retrievers), exploiting the fact that ranking is locally computable.
- Correctness relies on temporal and spatial locality of retrieved documents: the same or adjacent corpus entries tend to appear repeatedly in consecutive retrieval steps during a single generation request.
- After `s` speculative steps (speculation stride), a single batched knowledge-base query verifies all `s` results; the first mismatch triggers rollback and regeneration from that position.
- Cache can be updated with top-`k` results per verification batch (prefetching), improving speculation accuracy at the cost of higher per-batch retrieval overhead.
- Analogous to speculative execution in computer architecture and speculative decoding for LLMs, but applied to the retrieval axis rather than the decoding axis.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-accelerating-2401-14021]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-accelerating-2401-14021]].
