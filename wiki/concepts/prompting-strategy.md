---
type: concept
title: Prompting Strategy
slug: prompting-strategy
date: 2026-04-20
updated: 2026-04-20
aliases: [prompting policy, 提示策略]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompting Strategy** (提示策略) — an explicit prompting policy or workflow used to shape how a language model reasons, acts, and terminates on a task.

## Key Points

- [[yang-2023-intercode-2306-14898]] compares Single Turn, Try Again, ReAct, and Plan & Solve under the same interactive coding benchmark.
- The paper shows that different strategies expose different capabilities, including error correction, context discovery, and modular planning.
- ReAct substantially improves SQL success and admissibility, while rigid planning strategies can underperform when fast adaptation to feedback is needed.
- InterCode is positioned as a benchmark for studying not just models, but also how prompting policies interact with executable environments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-intercode-2306-14898]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-intercode-2306-14898]].
