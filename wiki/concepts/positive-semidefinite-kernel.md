---
type: concept
title: Positive Semidefinite Kernel
slug: positive-semidefinite-kernel
date: 2026-04-20
updated: 2026-04-20
aliases: [PSD kernel, 正半定核]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Positive Semidefinite Kernel** (正半定核) — a similarity function whose Gram matrix has nonnegative eigenvalues, ensuring that spectral quantities such as entropy are well-defined.

## Key Points

- The Vendi Score requires `k` to be positive semidefinite and normalized so that `k(x, x) = 1`.
- Under this assumption, the eigenvalues of `K / n` are nonnegative and sum to one, so Shannon entropy can be applied directly.
- The PSD requirement links the metric to kernel methods, covariance operators, and low-rank computational shortcuts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
