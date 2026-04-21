---
type: concept
title: Pseudo-Relevance Feedback
slug: pseudo-relevance-feedback
date: 2026-04-20
updated: 2026-04-20
aliases: [PRF, 伪相关反馈]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pseudo-Relevance Feedback** (伪相关反馈) — a retrieval strategy that treats top-ranked results as provisional positives and expands or reformulates the search using information derived from them.

## Key Points

- The paper uses LADR as a pseudo-relevance-feedback variant on top of BM25-first ColBERTv2 reranking.
- LADR expands the candidate set by visiting nearest neighbors of top-scoring documents, helping recover documents missed by pure lexical first-stage retrieval.
- The adaptive version repeatedly scores neighbors of the top `c in {10, 20, 50}` documents until convergence.
- On TREC DL 2019, this feedback-style expansion lets LADR dominate PLAID's Pareto frontier in both recall and nDCG.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[macavaney-2024-reproducibility]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[macavaney-2024-reproducibility]].
