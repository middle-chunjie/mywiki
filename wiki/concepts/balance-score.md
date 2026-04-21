---
type: concept
title: Balance Score
slug: balance-score
date: 2026-04-20
updated: 2026-04-20
aliases: [balance score, 平衡分数]
tags: [evaluation, reasoning, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Balance Score** (平衡分数) — a metric for self-improvement training that jointly rewards having enough unique correct responses and keeping a high proportion of selected responses correct.

## Key Points

- [[unknown-nd-bstar]] defines `bs_i = min(n_i' / n*, 1) * (n_i' / n_i)` for each query, where `n_i'` counts unique correct selected responses and `n_i` counts all selected responses.
- The `min(n_i' / n*, 1)` term caps the benefit of adding more correct samples so easy queries do not dominate the training set.
- The `n_i' / n_i` term measures cleanliness of the selected data and penalizes aggressive but noisy selection.
- B-STaR uses average balance score on a small subset of queries to choose temperature and reward threshold at every iteration.
- Reported balance score rises steadily across math training, from `0.470` at step `500` to `0.679` at step `4500`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-bstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-bstar]].
