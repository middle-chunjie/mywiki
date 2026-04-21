---
type: concept
title: Standard Score
slug: standard-score
date: 2026-04-20
updated: 2026-04-20
aliases: [standardized score]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Standard Score** (标准分) — a normalized score computed from a task's mean and standard deviation so different tasks can be compared on a common scale.

## Key Points

- KoLA first computes `z_ij = (x_ij - mu_i) / sigma_i` for each model-task pair, using the evaluated model pool for that task.
- It then applies Min-Max scaling to map the resulting z-scores into `[0, 100]`, which the benchmark uses for cross-task comparisons on the leaderboard.
- The paper pairs evolving and known tasks within each level before averaging, so the overall ranking depends on these standardized rather than raw task metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
