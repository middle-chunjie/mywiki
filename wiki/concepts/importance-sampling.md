---
type: concept
title: Importance Sampling
slug: importance-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [重要性采样]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Importance Sampling** (重要性采样) — a Monte Carlo technique that reweights samples drawn from one distribution so they estimate expectations under another distribution.

## Key Points

- The paper interprets Soft-InfoNCE weights as correcting the mismatch between uniformly sampled in-batch negatives and the real relevance distribution over negatives.
- Under this view, `w_ij` functions like a density-ratio correction term that reduces bias in estimating the negative expectation inside InfoNCE.
- The derivation argues that more truly negative samples should receive larger effective weight than near-positive negatives.
- This interpretation is central to the paper's claim that Soft-InfoNCE gives a more faithful mutual-information estimate than vanilla InfoNCE.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-rethinking-2310-08069]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-rethinking-2310-08069]].
