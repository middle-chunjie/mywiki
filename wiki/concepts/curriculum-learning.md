---
type: concept
title: Curriculum Learning
slug: curriculum-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [staged fine-tuning, 课程学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Curriculum Learning** (课程学习) — a training strategy that orders data or tasks in stages so the model first learns from one distribution and is then adapted to another.

## Key Points

- [[cui-2022-codeexp-2211-15395]] uses a two-stage curriculum that fine-tunes on CodeExp(raw) before continuing on CodeExp(refined).
- The staged recipe is motivated by combining broad coverage from noisy large-scale data with precision from a much smaller high-quality subset.
- Across the evaluated backbones, the raw-then-refined strategy gives the strongest automatic-metric performance.
- The paper treats the refined-only setting as strong, but shows the curriculum often yields the best final checkpoints.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
