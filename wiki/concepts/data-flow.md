---
type: concept
title: Data Flow
slug: data-flow
date: 2026-04-20
updated: 2026-04-20
aliases: [数据流]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Data Flow** (数据流) — the propagation of values through variable definitions, uses, and dependencies across a program.

## Key Points

- [[hooda-2024-do-2402-05980]] evaluates data-flow understanding with Independent Swap and Def-Use Break mutations.
- Def-Use Break explicitly renames a second def-use chain and its downstream uses to test whether the model respects scope and liveness.
- Data-flow-related degradations remain substantial, reaching `34.07%` on CodeContests for Llama 2 `7B`.
- The paper treats data-flow sensitivity as distinct from surface naming effects, even though both involve variable-level perturbations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
