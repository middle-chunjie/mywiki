---
type: concept
title: Model Parameter Scaling
slug: model-parameter-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [parameter scaling, scaling model parameters, 模型参数扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model Parameter Scaling** (模型参数扩展) — increasing a model's parameter count to improve capability, typically at higher pretraining and inference cost.

## Key Points

- The paper compares parameter scaling directly against adaptive test-time compute allocation.
- Its FLOPs-matched analysis fixes pretraining data and treats larger parameter count as the main way to spend extra training compute.
- A model with about `14x` more parameters remains preferable on the hardest questions and in high-inference-load regimes.
- On easier questions, the paper shows that targeted test-time compute can outperform this parameter scaling baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
