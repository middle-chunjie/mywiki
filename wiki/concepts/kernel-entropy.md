---
type: concept
title: Kernel Entropy
slug: kernel-entropy
date: 2026-04-20
updated: 2026-04-20
aliases: [核熵]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Kernel Entropy** (核熵) — an entropy computed from the spectrum of a normalized kernel matrix, measuring uncertainty or spread in the similarity structure of a sample.

## Key Points

- The Vendi Score is the exponential of kernel entropy, with entropy taken over the eigenvalues of `K / n`.
- Kernel entropy lets the paper define diversity directly from pairwise similarities instead of requiring a tractable sample distribution.
- The appendix cites a `1 / sqrt(n)` convergence rate for empirical kernel-entropy estimation, which transfers to the Vendi Score estimator.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
