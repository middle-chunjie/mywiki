---
type: concept
title: Brier Score
slug: brier-score
date: 2026-04-20
updated: 2026-04-20
aliases: [Brier score, 布里尔分数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Brier Score** (布里尔分数) — a strictly proper scoring rule that measures the squared error between predicted probabilities and observed outcomes for mutually exclusive events.

## Key Points

- [[schaeffer-2023-emergent-2304-15004]] uses Brier Score as a continuous alternative to Multiple Choice Grade in its LaMDA re-analysis.
- The paper shows that tasks appearing emergent under Multiple Choice Grade lose that pattern when rescored with Brier Score.
- This supports the paper's claim that discontinuous evaluation metrics can create the illusion of abrupt capability jumps.
- The concept matters because it preserves probabilistic gradation instead of collapsing outputs into a binary pass/fail label.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[schaeffer-2023-emergent-2304-15004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[schaeffer-2023-emergent-2304-15004]].
