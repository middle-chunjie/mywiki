---
type: concept
title: Data-Centric Optimization
slug: data-centric-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data-Centric Optimization** — improving system behavior by changing prompts, data organization, or output structure instead of changing model parameters, system kernels, or hardware.

## Key Points

- The paper positions SoT as a data-level efficiency method rather than a model- or system-level optimization.
- The key intervention is to reorganize output content into a short skeleton plus independent expansions.
- This view treats answer structure as a controllable resource for latency reduction.
- The paper argues the approach becomes more viable as instruction-following and planning abilities of LLMs improve.
- Prompt shortening and KV-cache reuse are suggested as future ways to reduce SoT overhead within this paradigm.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ning-2024-skeletonofthought-2307-15337]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ning-2024-skeletonofthought-2307-15337]].
