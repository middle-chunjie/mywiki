---
type: concept
title: Hidden Markov Model
slug: hidden-markov-model
date: 2026-04-20
updated: 2026-04-20
aliases: [HMM, hidden markov model, 隐马尔可夫模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hidden Markov Model** (隐马尔可夫模型) — a probabilistic sequence model with latent states and observed emissions, where transitions and emissions define the distribution over observation sequences.

## Key Points

- The paper uses HMMs as the theoretical bridge for reasoning about how conditions alter sequence distributions.
- A conditioned HMM state is decomposed into semantic and condition subspaces, `φ_s = [φ_{s,semantic}; φ_{s,condition}]`.
- Under normalization and independence assumptions, changing the condition part of the initial state induces a linear transform on the equivalent LM embedding space.
- This derivation is the formal motivation for LM-Switch rather than a direct training procedure used in experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
