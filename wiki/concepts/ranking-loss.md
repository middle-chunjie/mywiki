---
type: concept
title: Ranking Loss
slug: ranking-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [pairwise ranking loss, 排序损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Ranking loss** (排序损失) — an objective that enforces the relative ordering of candidate outputs, pushing higher-quality responses to receive higher model scores than lower-quality ones.

## Key Points

- RRHF instantiates ranking loss as `L_rank = sum_{r_i < r_j} max(0, p_i - p_j)`, where `p_i` is the model score and `r_i` is the preference or reward score.
- The loss operates on comparisons among multiple responses to the same query rather than on absolute reward magnitudes.
- In this paper, ranking loss is combined with a supervised fine-tuning term on the best response instead of replacing likelihood training entirely.
- The authors report that removing `L_rank` worsens reward performance from `-1.03` to `-1.14` in their Alpaca beam-search setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2023-rrhf-2304-05302]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2023-rrhf-2304-05302]].
