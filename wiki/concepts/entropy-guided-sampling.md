---
type: concept
title: Entropy-Guided Sampling
slug: entropy-guided-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [熵引导采样]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Entropy-Guided Sampling** (熵引导采样) — a trajectory-construction method that branches generation from high-entropy reasoning positions to obtain diverse preference pairs more efficiently than repeated full-path rollouts.

## Key Points

- Tool-Light first generates a main chain, then computes average entropy over the first `10`, `20`, `30`, `40`, and `50` tokens of each step.
- It selects the top-`k` highest-entropy steps as branch points and continues sampling from those positions.
- In the idealized analysis, this reduces rollout cost from `O(mn)` to `O(n log m)` for `m` samples of average length `n`.
- The resulting branch chains are mixed with vanilla rollouts, and Appendix B reports a final data ratio of `13:7`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
