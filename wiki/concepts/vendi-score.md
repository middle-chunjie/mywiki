---
type: concept
title: Vendi Score
slug: vendi-score
date: 2026-04-20
updated: 2026-04-20
aliases: [VS]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Vendi Score** — a diversity metric defined as the exponential of the Shannon entropy of the eigenvalues of a normalized similarity matrix, interpretable as the effective number of distinct elements in a sample.

## Key Points

- The paper defines `VS_k(x_1, ..., x_n) = exp(-sum_i lambda_i log lambda_i)` where `lambda_i` are eigenvalues of `K / n`.
- Vendi Score is reference-free: it only needs the evaluated sample and a user-chosen similarity function.
- The score equals `1` for completely identical samples and `n` for `n` mutually dissimilar samples.
- Experiments show it can detect duplicate clusters and subtle internal redundancy that IntDiv and number-of-modes metrics miss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
