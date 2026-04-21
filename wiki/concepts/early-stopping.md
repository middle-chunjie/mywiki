---
type: concept
title: Early Stopping
slug: early-stopping
date: 2026-04-20
updated: 2026-04-20
aliases: [early stopping, 早停]
tags: [optimization, regularization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Early Stopping** (早停) — terminating training before convergence when validation performance stops improving in order to reduce overfitting or improve compute efficiency.

## Key Points

- Kaplan et al. use early stopping when studying finite-dataset scaling so that dataset-limited runs stop once test loss ceases to improve.
- Their joint law `L(N, D)` treats early-stopped loss as the relevant quantity for analyzing overfitting under finite data.
- The paper derives a lower bound for the stopping point, `S_stop(N, D) \gtrsim S_c / [L(N, D) - L(N, \infty)]^{1 / alpha_S}`.
- Compute-efficient training is intentionally stopped well before convergence, and the appendix argues that this can save substantial compute at fixed target loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kaplan-2020-scaling-2001-08361]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kaplan-2020-scaling-2001-08361]].
