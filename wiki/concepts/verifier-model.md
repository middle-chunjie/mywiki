---
type: concept
title: Verifier Model
slug: verifier-model
date: 2026-04-20
updated: 2026-04-20
aliases: [verifier, verifier-based scoring, 验证器模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Verifier Model** (验证器模型) — a learned scoring model that evaluates candidate solutions and guides answer selection or search at inference time.

## Key Points

- The paper treats the verifier as one of the two central levers for scaling test-time compute.
- Its main verifier is a [[process-reward-model]] that scores intermediate reasoning steps.
- Verifier outputs are aggregated with best-of-`N` weighted selection to choose final answers.
- Stronger optimization against a verifier can help on harder questions but can overfit spurious signals on easy ones.
- The paper also uses an ORM-style verifier for revision trajectories when the base PRM suffers from distribution shift.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
