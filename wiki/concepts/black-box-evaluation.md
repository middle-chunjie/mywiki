---
type: concept
title: Black-Box Evaluation
slug: black-box-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [hard-label evaluation, 黑盒评估]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Evaluation** (黑盒评估) — evaluation that treats the model as an input-output system and does not require access to internal states, gradients, or token probabilities.

## Key Points

- [[hooda-2024-do-2402-05980]] explicitly targets hard-label black-box access, making the method usable for closed models.
- The framework evaluates full generated outputs rather than only the next-token distribution.
- This distinguishes the paper from prior code-model counterfactual work that depends on output probabilities.
- Black-box access makes the method practical for widely deployed API-based code models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
