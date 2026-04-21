---
type: concept
title: Candidate Sampling
slug: candidate-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [candidate generation, 候选点采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Candidate Sampling** (候选点采样) — the process of generating prospective input configurations that will later be scored and filtered by an acquisition rule.

## Key Points

- LLAMBO samples candidates conditionally from `p(h | s'; D_n)` instead of only sampling from a generic search prior.
- The target score is set by `s' = s_min - alpha * (s_max - s_min)`, making exploration explicitly controllable through `alpha`.
- The approach is inspired by TPE but aims to target a desired objective value directly rather than merely separating good and bad regions.
- The paper reports a trade-off: stronger candidate quality can reduce diversity, especially as `alpha` increases.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
