---
type: concept
title: Self-Evolution
slug: self-evolution
date: 2026-04-20
updated: 2026-04-20
aliases: [self-evolved learning, 自演化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Evolution** (自演化) — an iterative training strategy in which a model generates new data from its current behavior and then uses that data to further update itself.

## Key Points

- Tool-Light uses the current DPO-trained model to resample trajectories and build the next round of preference data.
- The easy and hard sets are redefined after each round so that the data difficulty follows the model's evolving capability.
- This mechanism is intended to preserve efficient tool use while teaching the model to make necessary tool calls on harder cases.
- The paper's ablation shows that self-evolution helps up to `2` loops, after which performance declines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
