---
type: concept
title: Sub-Goal Selection
slug: sub-goal-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [subgoal selection, sub-goal routing]
tags: [planning, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sub-Goal Selection** (子目标选择) — the decision step that chooses which high-level operation should be executed next in a multi-step reasoning process.

## Key Points

- In CR-Planner, sub-goal selection routes the system among `Reason`, `GenQuery`, and `Retrieve`.
- The decision is made by a dedicated critic `g^g` that scores each available sub-goal under the current trajectory state.
- Because retrieval is optional, the model can postpone or skip retrieval when reasoning alone appears more promising.
- The paper treats sub-goal choice as a planning decision, not as a fixed template baked into prompting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
