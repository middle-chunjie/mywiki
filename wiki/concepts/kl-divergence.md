---
type: concept
title: KL Divergence
slug: kl-divergence
date: 2026-04-20
updated: 2026-04-20
aliases: [Kullback-Leibler divergence, KL 散度]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**KL Divergence** (KL 散度) — a measure of how far one probability distribution deviates from another reference distribution.

## Key Points

- The paper proves that Soft-InfoNCE upper-bounds an objective containing a KL term between the target negative-similarity distribution `S_i` and the model distribution `P_theta(c_j | q_i)`.
- This interpretation explains Soft-InfoNCE as shaping negative-pair similarity distributions rather than merely suppressing all negatives uniformly.
- The authors compare this implicit KL control against explicitly adding KL regularization to InfoNCE and find the explicit regularizer performs worse.
- Hyperparameters `α` and `β` govern the tradeoff between minimizing negative similarity and matching the target distribution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-rethinking-2310-08069]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-rethinking-2310-08069]].
