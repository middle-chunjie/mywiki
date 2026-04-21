---
type: concept
title: Pretraining Cooldown
slug: pretraining-cooldown
date: 2026-04-20
updated: 2026-04-20
aliases: [cooldown training, 预训练冷却阶段]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pretraining Cooldown** (预训练冷却阶段) — a short final phase of pretraining that up-weights higher-quality data and anneals the learning rate to improve downstream behavior before training stops.

## Key Points

- [[unknown-nd-code]] applies cooldown for `40B` tokens, or `10%` of the earlier pretraining budget.
- The cooldown mixture emphasizes higher-quality text, math, code, and instruct-style data.
- The learning-rate schedule changes from cosine decay to linear annealing with final learning rate `1e-6`.
- Including `20%` code during cooldown improves reasoning, world knowledge, code performance, and generative win-rates relative to cooldown without code.
- The paper argues that code remains useful even in the final, high-quality refinement stage rather than only in early pretraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code]].
