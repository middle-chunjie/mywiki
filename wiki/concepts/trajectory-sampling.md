---
type: concept
title: Trajectory Sampling
slug: trajectory-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [trajectory sampling]
tags: [agents, llm, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trajectory Sampling** — the process of generating multiple candidate reasoning trajectories for the same input in order to improve coverage, diversity, or downstream training quality.

## Key Points

- [[gou-2024-tora-2309-17452]] uses sampling in two places: first to recover valid GPT-4 annotations when greedy decoding fails, and later to enlarge the training output space of ToRA.
- During corpus construction, the paper applies nucleus sampling with sample size `10` and keeps up to `4` valid trajectories per question.
- During output space shaping, the imitation-learned model samples `64` trajectories per training question before filtering and teacher correction.
- Sampling materially raises coverage on MATH, where the final annotation success rate reaches `83.1%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2024-tora-2309-17452]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2024-tora-2309-17452]].
