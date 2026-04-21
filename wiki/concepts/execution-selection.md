---
type: concept
title: Execution Selection
slug: execution-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [candidate execution selection, execution scoring]
tags: [planning, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Execution Selection** (执行选择) — the step that ranks sampled candidate realizations of a chosen sub-goal and keeps the one estimated to yield the best future outcome.

## Key Points

- After CR-Planner selects a sub-goal, it samples several rationales, queries, or retrieved documents as candidate executions.
- Separate execution critics score these candidates for rationale, query, and document states.
- This decouples exploration from commitment: the generator can sample broadly while the critic filters toward higher-value continuations.
- The paper uses `3` samples per execution during inference to balance accuracy and cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
