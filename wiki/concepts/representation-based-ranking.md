---
type: concept
title: Representation-Based Ranking
slug: representation-based-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [representation-based ranker, 表征式排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Representation-Based Ranking** (表征式排序) — a ranking paradigm that encodes query and document separately into global representations and scores relevance from their embedding similarity.

## Key Points

- In this paper, `BERT (Rep)(q, d) = cos(q_cls^last, d_cls^last)` is the concrete representation-based baseline.
- Separate encoding removes the query-document cross attention that makes BERT effective in reranking, so the model behaves poorly despite strong language-model pretraining.
- The approach performs far worse than interaction-based BERT on both MS MARCO and ClueWeb, showing that pretrained `[CLS]` embeddings are not sufficient as standalone retrieval representations here.
- The comparison sharpens the paper's claim that BERT should be treated primarily as a joint matching model for these tasks rather than as a bi-encoder-style retriever.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qiao-2019-understanding-1904-07531]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qiao-2019-understanding-1904-07531]].
