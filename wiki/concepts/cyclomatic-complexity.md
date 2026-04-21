---
type: concept
title: Cyclomatic Complexity
slug: cyclomatic-complexity
date: 2026-04-20
updated: 2026-04-20
aliases: [圈复杂度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cyclomatic complexity** (圈复杂度) — a graph-based measure of control-flow complexity that counts the number of linearly independent execution paths in a program.

## Key Points

- The paper augments validity and coverage rewards with normalized cyclomatic complexity in `R_4`.
- This term is intended to bias generation toward programs that induce richer compiler control flow and optimization behavior.
- Sample synthesized programs under `Reward 4` increase from cyclomatic complexity `2` to `39` as training proceeds.
- The reported average program complexity across reward settings grows from `4` to `18`, tracking the observed coverage improvements.
- Adding this signal improves final coverage relative to `Reward 3`, but it also lowers validity compared with the most validity-focused reward designs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
