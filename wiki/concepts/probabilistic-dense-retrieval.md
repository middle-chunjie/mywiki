---
type: concept
title: Probabilistic Dense Retrieval
slug: probabilistic-dense-retrieval
date: 2026-04-20
aliases: [probabilistic retrieval representation, uncertainty-aware dense retrieval]
tags: [dense-retrieval, probabilistic-modeling, information-retrieval, uncertainty]
source_count: 1
confidence: low
---

## Definition

**Probabilistic Dense Retrieval** — a class of neural retrieval models that represent queries and/or documents as probability distributions rather than point vectors, enabling the retrieval system to model and exploit uncertainty or confidence in the learned representations during scoring.

## Key Points

- Classical dense retrieval (DPR, TAS-B, ANCE) uses fixed vectors; probabilistic extensions assign a distribution to each input so that high-variance representations indicate ambiguous or polysemous inputs.
- MRL [[zamani-2023-multivariate-2304-14522]] instantiates this with multivariate normals and negative KL divergence scoring, whereas earlier Bayesian approaches (Cohen et al. 2021) used Monte Carlo dropout but were limited to reranking.
- Scoring functions based on distributional divergence (KL, Wasserstein) generalize dot-product similarity: when all variances collapse to a constant, KL-divergence ranking reduces to Euclidean distance ranking.
- A key practical challenge is compatibility with ANN indexes that only support dot products; MRL solves this by reformulating KL divergence into a dot product between augmented `(2k+2)`-dimensional vectors.
- Variance norms of query representations carry an emergent signal about query difficulty, linking this framework to query performance prediction literature.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2023-multivariate-2304-14522]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2023-multivariate-2304-14522]].
