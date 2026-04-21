---
type: entity
title: TREC DL 2019
slug: trec-dl-2019
date: 2026-04-20
entity_type: dataset
aliases: [TREC Deep Learning 2019]
tags: []
---

## Description

TREC DL 2019 is the densely judged deep-learning retrieval benchmark used in [[macavaney-2024-reproducibility]] to complement the sparse MS MARCO Dev evaluation. It provides `43` queries with roughly `215` assessments per query, making deep-ranking quality and recall easier to interpret.

## Key Contributions

- Reveals stronger differences in `nDCG@1k` and `R@1k` across PLAID operating points than MS MARCO Dev does.
- Shows that LADR can surpass PLAID on both recall and nDCG when dense judgments are available.

## Related Concepts

- [[late-interaction]]
- [[reranking]]
- [[rank-biased-overlap]]

## Sources

- [[macavaney-2024-reproducibility]]
