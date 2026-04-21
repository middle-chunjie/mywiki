---
type: concept
title: Speculative Sampling
slug: speculative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [投机采样]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Speculative Sampling** (投机采样) — an exact stochastic sampling procedure that draws from a cheap proposal distribution and corrects rejected proposals with a residual distribution so the final sample still follows the target distribution.

## Key Points

- The paper samples a draft token from `q(x)` and accepts it with probability `min(1, p(x)/q(x))`.
- If the draft token is rejected, the correction is sampled from ``norm(max(0, p(x) - q(x)))`` rather than from unmodified `p(x)`.
- This construction guarantees that the final token distribution is exactly `p(x)` for any proposal model `q`.
- The acceptance probability is ``β = Σ_x min(p(x), q(x))``, connecting exactness to overlap between target and proposal distributions.
- Speculative decoding generalizes this token-level sampler to `γ` drafted positions verified in parallel.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[leviathan-2023-fast-2211-17192]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[leviathan-2023-fast-2211-17192]].
