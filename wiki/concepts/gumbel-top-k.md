---
type: concept
title: Gumbel-Top-k
slug: gumbel-top-k
date: 2026-04-20
updated: 2026-04-20
aliases: [Gumbel top-k, gumbel top k, Ancestral Gumbel-Top-k]
tags: [optimization, sampling, retrieval, discrete-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gumbel-Top-k** — a technique for sampling k items without replacement from a categorical distribution by independently perturbing each item's log-probability with Gumbel noise and selecting the top-k perturbed values.

## Key Points

- Independently drawing `G_d ~ -log(-log(U))`, where `U ~ Uniform(0,1)`, and selecting the `k` items with largest `s^φ_{xd} + G_d` produces a valid sample from the Plackett-Luce distribution (Kool et al., 2019).
- The straight-through variant used in [[zamani-2024-stochastic]] applies argmax (top-k) in the forward pass but routes gradients through the softmax distribution in the backward pass, making the selection differentiable.
- This approach extends [[gumbel-softmax]] from single-item sampling to ordered k-item sampling without replacement, enabling end-to-end optimization of retrieval in [[retrieval-augmented-generation]] systems.
- Prior work applied Gumbel-top-k to re-ranking models conditioned on first-stage retrieval (Zamani et al., 2022a); [[zamani-2024-stochastic]] extends it to the full RAG objective including generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2024-stochastic]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2024-stochastic]].
