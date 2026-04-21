---
type: concept
title: Hidden Intention
slug: hidden-intention
date: 2026-04-20
updated: 2026-04-20
aliases: [隐含意图, implicit intent]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hidden Intention** (隐含意图) — user intent or required task information that is omitted from the current turn and must be inferred from prior dialogue context.

## Key Points

- WildToolBench models hidden intention through partial information, coreference, and long-range dependency settings.
- The benchmark treats cross-turn inference as a central agent capability, not a side condition.
- Long-range dependency is the hardest hidden-intention subtype in the paper, with no model exceeding `50%` accuracy.
- Reasoning-oriented models tend to do better on omitted-information recovery than non-reasoning counterparts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-benchmarking-2604-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-benchmarking-2604-06185]].
