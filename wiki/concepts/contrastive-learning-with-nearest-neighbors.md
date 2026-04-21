---
type: concept
title: Contrastive Learning with Nearest Neighbors
slug: contrastive-learning-with-nearest-neighbors
date: 2026-04-20
updated: 2026-04-20
aliases: [CLNN, Neighborhood Contrastive Learning]
tags: [contrastive-learning, clustering, representation-learning, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Learning with Nearest Neighbors** (CLNN) — a self-supervised clustering method that defines positive pairs in the contrastive loss using mined nearest-neighbor sets in the embedding space, replacing the conventional single-augmentation-pair positive definition and thereby avoiding pushing away false negatives.

## Key Points

- For each training instance `x_i`, a top-`K` nearest-neighbor set `N_i` is precomputed; one neighbor is sampled per step and augmented to form a positive pair alongside the augmented anchor.
- The generalized contrastive loss `l_i = -(1/|C_i|) Σ_{j∈C_i} log [exp(sim/τ) / Σ_{k≠i} exp(sim/τ)]` averages over all positive pairs in `C_i`, including both the neighbor-based pairs and, in semi-supervised settings, known-intent pairs.
- Neighborhoods are updated periodically (every 5 epochs) as the embedding space evolves during training.
- Optimal neighborhood size `K` is approximately half the average per-class training set size; this empirical rule transfers across datasets.
- Directly optimizes in the feature space rather than over clustering logits (as in SCAN), which has been shown to yield more effective representations.
- Data augmentation by random token replacement (RTR, probability 0.25) outperforms EDA, token shuffling, and dropout augmentation for short intent utterances.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-new-2205-12914]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-new-2205-12914]].
