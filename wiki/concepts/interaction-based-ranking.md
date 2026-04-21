---
type: concept
title: Interaction-Based Ranking
slug: interaction-based-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [interaction-based ranker, 交互式排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Interaction-Based Ranking** (交互式排序) — a ranking paradigm that scores a query-document pair from token-level or sequence-level cross interactions computed jointly before producing a final relevance score.

## Key Points

- This paper operationalizes the concept with `BERT (Last-Int)`, which concatenates query and document using `[SEP]` and scores relevance from the final joint `[CLS]` embedding.
- Cross query-document attention is the main source of BERT's gains: removing joint encoding and using separate representations collapses effectiveness on both benchmarks.
- On MS MARCO, interaction-based BERT variants outperform earlier neural rerankers such as Conv-KNRM by large margins, showing that pretrained Transformer interactions transfer well to QA-style passage ranking.
- More elaborate interaction architectures, including layer-wise feature aggregation and translation-style token matching, do not beat the simplest last-layer interaction setup.
- The paper argues that BERT's interaction behavior resembles sequence-to-sequence semantic matching more than click-trained relevance matching for ad hoc search.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qiao-2019-understanding-1904-07531]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qiao-2019-understanding-1904-07531]].
