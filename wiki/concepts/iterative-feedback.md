---
type: concept
title: Iterative Feedback
slug: iterative-feedback
date: 2026-04-20
updated: 2026-04-20
aliases: [feedback iteration, iterative refinement feedback]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Iterative Feedback** (迭代反馈) — a retrieval-enhancement procedure in which an LLM repeatedly critiques current results and produces improved inputs for the next retrieval round.

## Key Points

- The paper decomposes feedback into comprehension, assessment, and refinement rather than asking for a single opaque rewrite.
- Each round conditions on both the current instruction and the current top-`K` retrieved tools, so the feedback is retrieval-state dependent.
- The LLM may terminate the loop by emitting `N/A` when it judges that no further refinement is needed.
- Effectiveness increases across `T = 1, 2, 3` rounds, with `NDCG@1` rising from `85.69` to `87.78` to `89.01` in the `I2` setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-enhancing-2406-17465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-enhancing-2406-17465]].
