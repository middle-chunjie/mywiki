---
type: concept
title: Tactic Prediction
slug: tactic-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [next-tactic prediction]
tags: [theorem-proving, generation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tactic Prediction** — the problem of generating the next formal proof command from the current proof state, optionally with additional latent reasoning variables.

## Key Points

- [[unknown-nd-leanstar]] starts from a direct tactic predictor `\pi_M(a | s)` trained on successful `(state, tactic)` pairs.
- Lean-STaR augments tactic prediction with a latent thought `t_i`, yielding `\pi_M(a_i, t_i | s_i) = \pi_M(a_i | t_i, s_i)\pi_M(t_i | s_i)`.
- The paper argues that predicting thoughts before tactics improves the model's ability to choose effective proof actions.
- Performance gains over both few-shot and SFT baselines suggest that the thought-augmented tactic predictor is more effective than direct prediction alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
