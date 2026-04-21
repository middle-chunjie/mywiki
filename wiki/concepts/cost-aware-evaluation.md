---
type: concept
title: Cost-Aware Evaluation
slug: cost-aware-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [成本感知评测, cost-sensitive evaluation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cost-Aware Evaluation** (成本感知评测) — an evaluation protocol that measures model or agent quality together with the monetary cost required to achieve it.

## Key Points

- AstaBench reports both score and normalized dollar cost because extra compute can artificially boost agent performance.
- The `agent-eval` layer maps Inspect usage logs to prices using a frozen LiteLLM cost snapshot to keep comparisons time-invariant.
- Cache discounts are included in reported cost, while latency-tier discounts are excluded.
- The leaderboard highlights Pareto-optimal quality-cost tradeoffs instead of ranking systems by accuracy alone.
- Openness and tooling labels are reported alongside cost so hidden infrastructure advantages are treated as confounders.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
