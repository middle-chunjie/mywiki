---
type: concept
title: Document Caching
slug: document-caching
date: 2026-04-20
updated: 2026-04-20
aliases: [doc caching, document caching, 文档缓存]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Caching** (文档缓存) — a retrieval-augmented generation strategy that stores document key-value states together with document embeddings so the generator can reuse them instead of recomputing document forward passes at inference time.

## Key Points

- GRIT introduces document caching as a natural consequence of using the same model for retrieval and generation.
- At indexing time, the system stores both document embeddings and key-value states; at inference time, retrieval returns cached states instead of raw text passages.
- On Natural Questions, the paper reports that document caching can improve match (`30.47` to `33.38`) while cutting CPU latency from `14.18s` to `5.25s` for a `4000`-token document case.
- The main trade-off is storage: for the reported `2,681,468`-document index, cached key-value states require about `30TB`.
- The paper argues document caching is the most promising caching variant because documents are usually longer than queries and therefore offer the largest latency savings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[muennighoff-2024-generative-2402-09906]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[muennighoff-2024-generative-2402-09906]].
