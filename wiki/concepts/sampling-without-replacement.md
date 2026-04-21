---
type: concept
title: Sampling Without Replacement
slug: sampling-without-replacement
date: 2026-04-20
updated: 2026-04-20
aliases: [sampling without replacement, ordered sampling, Plackett-Luce sampling]
tags: [probability, retrieval, optimization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sampling Without Replacement** — a stochastic process that sequentially draws items from a distribution where each selected item is removed from the pool, so the probability of subsequent draws is conditioned on prior selections.

## Key Points

- In [[zamani-2024-stochastic]], retrieval from a large collection is recast as sampling without replacement from the document score distribution, unifying the retrieval process with the generation objective.
- The probability of an ordered document list is computed via the Plackett-Luce chain rule: `p(d|x; R_φ) = Π_{i=1}^{k} p(d_i|x; R_φ) / (1 - Σ_{j<i} p(d_j|x; R_φ))`, where document-level probabilities come from a softmax over retrieval scores.
- This formulation explicitly models document dependence across positions, unlike the document-independence assumption in standard RAG that feeds each document independently to the generator.
- The sampling process is non-differentiable in its exact form; [[gumbel-top-k]] is used as a differentiable approximation to enable gradient-based end-to-end training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2024-stochastic]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2024-stochastic]].
