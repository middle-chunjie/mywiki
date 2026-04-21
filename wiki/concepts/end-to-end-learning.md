---
type: concept
title: End-to-End Learning
slug: end-to-end-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [端到端学习, end-to-end training]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**End-to-End Learning** (端到端学习) — a training setup in which intermediate representations are optimized jointly with the final task objective rather than fixed beforehand.

## Key Points

- K-NRM learns embeddings and ranking parameters together from pairwise ranking supervision.
- Differentiable kernel pooling makes it possible for gradients from the ranking loss to update token similarities and embeddings directly.
- The paper shows end-to-end learning is a major source of gains over variants that keep word2vec or click2vec embeddings fixed.
- This joint optimization is presented as the mechanism that turns generic embedding similarity into retrieval-specific soft-match structure.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2017-endtoend-1706-06613]].
