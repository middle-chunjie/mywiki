---
type: concept
title: Prompt Optimization
slug: prompt-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt search, 提示优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Optimization** (提示优化) — the process of improving a prompt by searching over its demonstrations, ordering, or other controllable text-space choices to increase task performance.

## Key Points

- This paper frames prompt optimization as combinatorial search over both demonstration selection and order.
- It avoids gradient-based prompt tuning and instead stays in the original text space, which is compatible with black-box LLM services.
- The proposed fairness objective provides a unified signal for local ranking and global greedy construction.
- The work contrasts exhaustive enumeration with `O(N)` and `O(N^2)` approximate search strategies that are much cheaper.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-fairnessguided-2303-13217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-fairnessguided-2303-13217]].
