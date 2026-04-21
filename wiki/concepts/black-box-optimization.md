---
type: concept
title: Black-Box Optimization
slug: black-box-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [black box optimization, 黑盒优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Optimization** (黑盒优化) — optimization where the system can evaluate candidate solutions but does not expose gradients or an analytic update rule.

## Key Points

- The paper hides the analytic linear-regression objective from the optimizer model so the task remains genuinely black-box.
- OPRO treats prompt optimization as black-box search because only task accuracy from a scorer LLM is observed.
- The optimization signal is external evaluation over candidate solutions, not parameter updates inside the optimizer LLM.
- The paper argues that natural-language descriptions let an LLM adapt to diverse black-box objectives without writing a custom solver.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-large-2309-03409]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-large-2309-03409]].
