---
type: concept
title: Human Preference Alignment
slug: human-preference-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [alignment to human preferences, 人类偏好对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Human preference alignment** (人类偏好对齐) — the process of tuning a model so that its outputs better match what human raters prefer in terms of helpfulness, harmlessness, correctness, or related criteria.

## Key Points

- RRHF frames preference alignment as ordering candidate responses by model probability so that this order matches human or reward-model preferences.
- The method is designed as a simpler alternative to PPO-based RLHF while preserving the ability to learn from pairwise or listwise response comparisons.
- Candidate responses can come from the model itself, stronger external systems such as ChatGPT or GPT-4, or human-authored data, broadening the supervision sources for alignment.
- The paper shows that alignment quality depends heavily on the quality of sampled responses, making data collection and sampling policy part of the alignment method itself.
- The authors also use RRHF with ChatGPT-based scoring to train Wombat, illustrating that preference alignment can be driven by external evaluators rather than only crowd-labeled datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2023-rrhf-2304-05302]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2023-rrhf-2304-05302]].
