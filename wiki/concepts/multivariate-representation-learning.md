---
type: concept
title: Multivariate Representation Learning
slug: multivariate-representation-learning
date: 2026-04-20
aliases: [MRL, multivariate representation]
tags: [dense-retrieval, probabilistic-modeling, representation-learning, information-retrieval]
source_count: 1
confidence: low
---

## Definition

**Multivariate Representation Learning (MRL)** — a representation learning framework for information retrieval that encodes each query and document as a multivariate probability distribution (specifically a multivariate normal) rather than a single vector, capturing both the central tendency and the uncertainty of the representation.

## Key Points

- Each query `q` and document `d` is represented as a `k`-variate normal distribution `N_k(M, Σ)` with a `k`-dim mean vector and a diagonal covariance matrix (i.e., `k`-dim variance vector), learned jointly via a bi-encoder (e.g., DistilBERT) with two special tokens `[CLS]` and `[VAR]`.
- Relevance scoring uses negative multivariate KL divergence: `score(q,d) = −KLD_k(Q ∥ D)`; for diagonal covariance this simplifies to a closed-form expression in terms of element-wise mean differences and variance ratios.
- The scoring function is algebraically reformulated into a dot product between augmented query and document vectors of dimension `2k+2`, enabling seamless integration with existing ANN libraries (FAISS/HNSW) without modifying the index structure.
- The learned variance (uncertainty) generalizes single-vector dense retrieval: when all documents share the same covariance, MRL reduces to Euclidean-distance-based ranking, making it a strict generalization.
- The norm of the query variance vector `|Σ_Q|` is an emergent, unsupervised signal that correlates with query retrieval performance, providing a built-in pre-retrieval query performance predictor.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2023-multivariate-2304-14522]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2023-multivariate-2304-14522]].
