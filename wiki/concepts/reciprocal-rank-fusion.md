---
type: concept
title: Reciprocal Rank Fusion
slug: reciprocal-rank-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [RRF, 倒数排序融合]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Reciprocal Rank Fusion** (倒数排序融合) — a rank aggregation method that combines multiple ranked lists by summing reciprocal rank scores for each candidate.

## Key Points

- [[lee-2024-gecko-2403-20327]] uses RRF to combine two LLM reranking signals: query likelihood and relevance classification.
- The fused ranking is `R(q,p) = 1 / r_QL(q,p) + 1 / r_RC(q,p)`, and Gecko uses that score to choose relabeled positives and negatives.
- In Appendix A, the fused reranker reaches `56.8` average nDCG@10 on BEIR, outperforming the baseline retriever and each single reranking prompt on average.
- The paper treats RRF as a robustness mechanism because the two prompting strategies perform differently across tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-gecko-2403-20327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-gecko-2403-20327]].
