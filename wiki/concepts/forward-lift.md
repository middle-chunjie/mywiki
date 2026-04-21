---
type: concept
title: Forward Lift
slug: forward-lift
date: 2026-04-20
updated: 2026-04-20
aliases: [Forward Lift, 前向提升]
tags: [evaluation, metric, llm]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Forward Lift** (前向提升) — the improvement in round-trip performance attributable to the information supplied by the forward model beyond an uninformative baseline prompt.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] defines lift as `L_M^{sim}(x) = RTC_sim(x) - E_{x_hat ~ M^{-1}(epsilon)} [sim(x_hat, x)]`.
- Positive lift indicates that the forward description helps the backward model recover the original semantics.
- Negative lift can indicate confusing or misleading forward generations.
- The paper reports higher average lift for Gemini Pro than Gemini Nano 2 in both synthesis and editing settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
