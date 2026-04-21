---
type: entity
title: LADR
slug: ladr
date: 2026-04-20
entity_type: tool
aliases: [Lexically Accelerated Dense Retrieval]
tags: []
---

## Description

LADR is the lexical-first ColBERTv2 retrieval variant used as a strong baseline in [[macavaney-2024-reproducibility]]. It augments BM25 reranking by expanding candidates through nearest-neighbor documents of top-scoring results.

## Key Contributions

- Adds adaptive neighbor expansion on top of BM25-first reranking.
- Dominates PLAID's Pareto frontier on TREC DL 2019 in both recall and nDCG.
- Still trails PLAID as an approximation to exhaustive ColBERTv2 search, with Dev `RBO` topping out around `0.96`.

## Related Concepts

- [[reranking]]
- [[pseudo-relevance-feedback]]
- [[approximate-nearest-neighbor-search]]

## Sources

- [[macavaney-2024-reproducibility]]
