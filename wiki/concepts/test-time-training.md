---
type: concept
title: Test-Time Training
slug: test-time-training
date: 2026-04-20
updated: 2026-04-20
aliases: [TTT, 测试时训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Test-Time Training** (测试时训练) — a learning procedure that updates part of a model on each unlabeled test instance using an auxiliary self-supervised objective before making the final prediction.

## Key Points

- The paper treats each test instance `X` as its own learning problem and adapts the encoder parameters `W` on that single instance before predicting the main-task output.
- The inner-loop objective is token-level reconstruction, `ell(W; X) = (1 / 2n) sum_i ||g(f(phi(x_i); W)) - x_i||^2`, with only the shared feature extractor updated at test time.
- In MTTT, test-time training is no longer driven by a hand-crafted auxiliary task alone; the outer loop learns the self-supervised task parameters so that adaptation improves supervised accuracy.
- Recasting a context window as a dataset lets the paper interpret attention-like computation as explicit learning inside the forward pass.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-learn-2310-13807]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-learn-2310-13807]].
