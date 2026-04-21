---
type: concept
title: Future-KL
slug: future-kl
date: 2026-04-20
updated: 2026-04-20
aliases: [Future KL, future KL]
tags: [reinforcement-learning, llm-reasoning, credit-assignment]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Future-KL** (未来 KL) — a discounted cumulative log-probability-shift signal that estimates how a token influences the future trajectory during policy optimization.

## Key Points

- FIPO defines `FutureKL_t = Σ_{k=t}^T Δlog p_k`, where `Δlog p_k` is the log-probability shift between the current and old policy.
- The paper interprets this sum as a sample-based estimate of future-horizon KL divergence, linking a current token to the subsequent chain of reasoning.
- A binary mask removes negative-advantage tokens whose importance ratios exceed the Dual-Clip threshold, preventing unstable outliers from dominating the estimate.
- A soft decay window with `γ = 2^(-1/τ)` and `τ = 32` emphasizes near-future effects while smoothly down-weighting distant tokens.
- The final influence weight is `clip(exp(FutureKL_t), 1 - ε_f_low, 1 + ε_f_high)`, which re-weights token advantages inside the policy objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2026-fipo-2603-19835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2026-fipo-2603-19835]].
