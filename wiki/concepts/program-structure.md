---
type: concept
title: Program Structure
slug: program-structure
date: 2026-04-20
updated: 2026-04-20
aliases: [control-flow structure, program structures, 程序结构]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Structure** (程序结构) — the control-flow organization of a program, typically expressed through sequential execution, branching, and looping constructs.

## Key Points

- The paper treats sequence, branch, and loop as the three basic structures from which code solutions can be composed.
- SCoT uses these structures explicitly in intermediate reasoning so the model plans in a code-like form before emitting source code.
- The ablation study shows that removing explicit basic structures causes a clear drop in `Pass@k`, indicating that structural control flow is useful prompt scaffolding.
- Branch and loop markers mainly help disambiguate execution scope that plain natural-language CoT often leaves underspecified.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structured-2305-06599]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structured-2305-06599]].
