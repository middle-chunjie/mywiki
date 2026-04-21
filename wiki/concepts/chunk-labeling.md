---
type: concept
title: Chunk Labeling
slug: chunk-labeling
date: 2026-04-20
updated: 2026-04-20
aliases: [token labeling, chunk tagging, labeler-tagger]
tags: [rag, retrieval, token-classification, multi-hop-qa]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chunk Labeling** (块标注) — a retrieval post-processing step in which each retrieved document chunk is analyzed to (a) identify informative tokens that partially answer the current query, and (b) tag the chunk as requiring further retrieval (`<Continue>`) or not (`<Terminate>`).

## Key Points

- EfficientRAG's Labeler+Tagger implements chunk labeling as two output heads on a shared DeBERTa-v3-large encoder: one token-level binary classifier (useful/not-useful) and one sequence-level binary classifier (`<Continue>`/`<Terminate>`).
- The token labeling head operates on the concatenation of the query and the chunk; labeled tokens represent bridging entities or relations needed for the next retrieval hop.
- The chunk tagging head uses mean-pooled sequence embeddings to decide whether a chunk provides enough information (allowing termination) or requires successor hops.
- `<Continue>`-tagged chunks are added to a candidate pool passed to the final LLM generator; `<Terminate>`-tagged chunks from irrelevant branches are discarded without spawning new queries.
- Training labels are synthesized via LLM-based decomposition and token-level annotation using SpaCy tokenization.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhuang-2024-efficientrag-2408-04259]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhuang-2024-efficientrag-2408-04259]].
