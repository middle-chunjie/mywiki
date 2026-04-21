---
type: concept
title: Embedding Lookup
slug: embedding-lookup
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 嵌入查表
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Embedding Lookup** (嵌入查表) — an operation that retrieves precomputed token vectors from an embedding matrix instead of running a deep encoder at inference time.

## Key Points

- LightRetriever caches dense query token vectors for the full vocabulary into a single embedding matrix before serving.
- Online dense query encoding becomes a lookup-plus-average computation over query tokens, with no Transformer layers on the critical path.
- The paper reports that this replacement reduces query encoding time from over `100 s` to about `0.04 s` on the large-model speed benchmark.
- Because query tokens are modeled independently during training, the resulting token vectors are cacheable without requiring cross-token interaction at serving time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2026-lightretriever-2505-12260]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2026-lightretriever-2505-12260]].
