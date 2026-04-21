---
type: concept
title: Length Controlled Policy Optimization
slug: length-controlled-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [LCPO, token-budget control]
tags: [llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Length Controlled Policy Optimization** (长度控制策略优化) — a reinforcement-learning objective that trains a reasoning model to satisfy prompt-specified output-length constraints while maximizing answer correctness.

## Key Points

- LCPO appends a target-length instruction such as "Think for `n_gold` tokens" to each training prompt.
- The exact-budget variant rewards correct answers and penalizes deviation from the requested length with `I(y = y_gold) - alpha * |n_gold - n_y|`.
- The max-budget variant uses a clipped multiplicative objective so correct answers with small budget overruns still receive nonzero reward.
- The paper applies LCPO to a `1.5B` reasoning model and obtains a controllable model family called L1.
- LCPO outperforms budget-forcing baselines across short and medium reasoning budgets and transfers to several OOD tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
