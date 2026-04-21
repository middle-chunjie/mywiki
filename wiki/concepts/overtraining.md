---
type: concept
title: Overtraining
slug: overtraining
date: 2026-04-20
updated: 2026-04-20
aliases: [过度训练]
tags: [llm, pretraining, efficiency]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Overtraining** (过度训练) — training a model on substantially more data than the compute-optimal regime would prescribe in order to obtain a smaller but more capable model.

## Key Points

- The paper presents overtraining as a response to rising inference cost in large compute-optimal language models.
- Overtrained small models can be attractive over a model's lifetime, but they require very large token budgets and long training runs.
- Distillation is positioned as a cheaper way to approximate the capability of small overtrained models.
- The paper's scaling-law analysis clarifies when distillation can replace or complement overtraining and when direct supervised pretraining is still better.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[busbridge-2025-distillation-2502-08606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[busbridge-2025-distillation-2502-08606]].
