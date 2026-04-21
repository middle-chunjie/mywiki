---
type: concept
title: Reward augmented maximum likelihood
slug: reward-augmented-maximum-likelihood
date: 2026-04-20
updated: 2026-04-20
aliases: [RAML, 奖励增强最大似然]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Reward augmented maximum likelihood** (奖励增强最大似然) — a training principle that replaces hard targets with reward-weighted soft targets so learning better reflects downstream evaluation preferences.

## Key Points

- GenRT uses a RAML-based soft criterion for truncation rather than hard binary labels.
- The paper derives cut and no-cut targets from the relative rewards of `TDCG@T` and `TDCG@(T + beta)`.
- This lets the model optimize truncation decisions smoothly with respect to retrieval utility instead of strict supervision at a single cut point.
- RAML is one of the components that couples the truncation head more tightly to the retrieval metric.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-listaware-2402-02764]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-listaware-2402-02764]].
