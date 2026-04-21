---
type: concept
title: Percentile Threshold
slug: percentile-threshold
date: 2026-04-20
updated: 2026-04-20
aliases: [percentile filtering, 百分位阈值]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Percentile Threshold** (百分位阈值) — a cutoff defined by a chosen percentile of scores, used to retain items whose scores are at or above that percentile.

## Key Points

- [[li-2023-compressing]] computes `I_p = np.percentile([I(u_0), ..., I(u_k)], p)` over lexical-unit self-information scores.
- Units are retained when `I(u_i) >= I_p`, so the effective pruning adapts to each input rather than using a global numeric cutoff.
- The paper evaluates reduction ratios `0.2`, `0.35`, `0.5`, `0.65`, and `0.8` to study the efficiency-quality trade-off.
- The authors note that the best percentile depends on the task and context, leaving threshold selection as an open practical problem.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
