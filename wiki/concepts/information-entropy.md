---
type: concept
title: Information Entropy
slug: information-entropy
date: 2026-04-20
updated: 2026-04-20
aliases: [entropy, 信息熵]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Information Entropy** (信息熵) — a token-level uncertainty measure over the model's next-token distribution, used here to analyze how reasoning trajectories evolve around tool calls.

## Key Points

- The paper computes per-position entropy as `H(i) = -sum_j P(y_{ji} | y_{<i}) log P(y_{ji} | y_{<i})`.
- After a tool result arrives, entropy tends to rise, fluctuate, and then fall sharply before the next tool call.
- For the same question, lower-entropy trajectories often correspond to correct solutions with fewer tool calls.
- These entropy observations motivate both branch selection during sampling and preference construction during DPO training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
