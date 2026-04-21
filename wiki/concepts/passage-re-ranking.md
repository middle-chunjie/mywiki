---
type: concept
title: Passage Re-ranking
slug: passage-re-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [passage reranking, 段落重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Passage Re-ranking** (段落重排序) — a second-stage ranking procedure that rescoring retrieved candidate passages with a more expressive model to improve the final relevance ordering.

## Key Points

- [[ren-2023-rocketqav-2110-07367]] models passage re-ranking with a cross-encoder that jointly encodes the query and each candidate passage.
- The paper argues that re-ranking should not be trained independently from retrieval because the reranker's relevance signals can improve the retriever.
- RocketQAv2 replaces the common pointwise or pairwise objective with a unified listwise formulation so the re-ranker and retriever can share distributions over the same candidate list.
- The reranker receives both supervised listwise cross-entropy and mutual-distillation signal from dynamic listwise distillation.
- On MSMARCO, the reranker reaches `MRR@10 = 41.9` when paired with the RocketQAv2 retriever over top-`50` candidates.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-rocketqav-2110-07367]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-rocketqav-2110-07367]].
