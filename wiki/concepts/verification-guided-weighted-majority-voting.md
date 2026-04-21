---
type: concept
title: Verification-Guided Weighted Majority Voting
slug: verification-guided-weighted-majority-voting
date: 2026-04-20
updated: 2026-04-20
aliases: [VW-voting, verification-weighted voting]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Verification-Guided Weighted Majority Voting** (验证引导加权多数投票) — an answer aggregation scheme that weights sampled solutions by their self-verification states instead of counting every candidate answer equally.

## Key Points

- Each sampled solution contributes an answer `a^i` and a verification state `v^i ∈ {True, Uncertain, False}`.
- Candidate answers are scored by summing counts under state-dependent weights `w_T`, `w_U`, and `w_F`, with the intended ordering `w_T > w_U > w_F`.
- When all weights are `1`, the method reduces to naive majority voting used in self-consistency style decoding.
- On MATH, combining CSV with this voting rule and `k = 16` sampled paths boosts GPT4-Code from `73.54%` to `84.32%`.
- The paper's ablation shows the framework is robust when `w_T > w_U >= w_F`, but can underperform naive voting when badly configured, e.g. `w_T = 0.5`, `w_U = 0.5`, `w_F = 1`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-solving-2603-03507]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-solving-2603-03507]].
