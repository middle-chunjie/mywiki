---
type: concept
title: Long-range Dependency
slug: long-range-dependency
date: 2026-04-20
updated: 2026-04-20
aliases: [长程依赖, long-context dependency]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-range Dependency** (长程依赖) — a case where solving the current task requires retrieving and integrating information that appeared many turns earlier in the dialogue.

## Key Points

- WildToolBench identifies long-range dependency as the hardest hidden-intention pattern in realistic tool-use conversations.
- No evaluated model exceeds `50%` accuracy on this subtask family.
- The paper highlights it as the dimension with the largest inter-model gap, around `17.3` points.
- Failures are linked to diluted attention over long dialogue context and interference from previous tool-use decisions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-benchmarking-2604-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-benchmarking-2604-06185]].
