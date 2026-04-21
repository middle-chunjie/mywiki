---
type: concept
title: Consistency Regularization
slug: consistency-regularization
date: 2026-04-20
updated: 2026-04-20
aliases: [consistency loss, 一致性正则化]
tags: [regularization, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Consistency Regularization** (一致性正则化) — a regularization strategy that encourages two stochastic forward passes of the same input to produce similar output distributions.

## Key Points

- The paper tests consistency loss from NCI as an auxiliary objective on top of generation loss.
- The regularizer symmetrizes two KL terms across dropout-perturbed output distributions: `1/2 [KL(p_{i,1} || p_{i,2}) + KL(p_{i,2} || p_{i,1})]`.
- In the reported experiments, consistency regularization often caused instability and `NaN` divergence.
- Because of that instability, the authors exclude it from their final large-scale setup and do not regard it as a reliable scaling ingredient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
