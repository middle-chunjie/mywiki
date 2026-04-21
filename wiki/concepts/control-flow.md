---
type: concept
title: Control Flow
slug: control-flow
date: 2026-04-20
updated: 2026-04-20
aliases: [控制流]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Control Flow** (控制流) — the ordering and branching structure that determines which program statements execute under which conditions.

## Key Points

- [[hooda-2024-do-2402-05980]] tests control-flow understanding primarily with the If-Else Flip mutation.
- The mutation negates relational expressions and swaps then/else bodies to preserve semantics while changing the local control predicate.
- If-Else Flip causes some of the largest observed AME values, often above `20%`, indicating weak robustness to control-flow perturbations.
- The Independent Swap mutation also touches control structure indirectly by reordering code when no dependency exists.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
